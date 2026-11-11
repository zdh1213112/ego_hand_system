#!/usr/bin/env python3
"""Instrument a WiLoR script in memory; never write to its source file.

The independent launcher substitutes this bootstrap only in the inference
subprocess. AST spans retain the original operations, line numbers and order;
module hooks provide the neural network breakdown.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

from wilor_timing import timer, timed


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)


def span(name, statements, gpu=False):
    result = ast.With(
        items=[ast.withitem(context_expr=ast.Call(
            func=ast.Attribute(value=ast.Name(id="_ego_timer", ctx=ast.Load()),
                               attr="span", ctx=ast.Load()),
            args=[ast.Constant(name)],
            keywords=[ast.keyword(arg="gpu", value=ast.Constant(gpu))]),
            optional_vars=None)], body=statements, type_comment=None)
    return ast.copy_location(result, statements[0])


def call_statement(code, location):
    return ast.copy_location(ast.parse(code).body[0], location)


class Instrument(ast.NodeTransformer):
    def __init__(self):
        self.functions = set()

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        name = node.name
        self.functions.add(name)
        timing_names = {
            "infer_dual": "inference.frame",
            "_prepare_dual_batched": "preprocess.chunk",
            "_infer_prepared_batched": "inference.chunk",
        }
        if name in timing_names:
            node.decorator_list.insert(0, ast.Call(
                func=ast.Name(id="_ego_timed", ctx=ast.Load()),
                args=[ast.Constant(timing_names[name])], keywords=[]))
        if name == "infer_dual":
            for index, statement in enumerate(node.body):
                if assignment(statement, "detection"):
                    node.body[index] = span("detector", [statement], True)
                if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name) and statement.target.id == "batch":
                    statement.iter = ast.Call(
                        func=ast.Attribute(value=ast.Name(id="_ego_timer", ctx=ast.Load()), attr="iterate", ctx=ast.Load()),
                        args=[ast.Constant("preprocess.crops"), statement.iter], keywords=[])
                    # Original loop: transfer -> inference context -> postprocessing.
                    assert assignment(statement.body[0], "batch")
                    assert isinstance(statement.body[1], ast.With)
                    statement.body = [span("transfer.h2d", statement.body[:1], True),
                                      statement.body[1], span("postprocess", statement.body[2:])]
        elif name == "_prepare_dual_batched":
            for index, statement in enumerate(node.body):
                if isinstance(statement, ast.If):
                    test = ast.unparse(statement.test)
                    if test == "external_boxes is None":
                        node.body[index] = span("detector", [statement], True)
                    elif test == "datasets":
                        node.body[index] = span("preprocess.crops", [statement])
        elif name == "_infer_prepared_batched":
            loop = next(item for item in node.body if isinstance(item, ast.While))
            attempt = next(item for item in loop.body if isinstance(item, ast.Try))
            assert assignment(attempt.body[0], "batch")
            attempt.body[0] = span("transfer.h2d", attempt.body[:1], True)
            inference = attempt.body[1]
            assert isinstance(inference, ast.With)
            inference.body = [inference.body[0], span("postprocess.camera", inference.body[1:], True)]
            start = next(index for index, item in enumerate(loop.body) if assignment(item, "rights"))
            end = next(index for index, item in enumerate(loop.body) if assignment(item, "batch_start"))
            loop.body[start:end] = [span("postprocess", loop.body[start:end])]
        elif name == "load" and any(assignment(item, "frames") for item in node.body):
            node.body = [span("video.decode", [item]) if assignment(item, "frames") else item
                         for item in node.body]
        elif name == "main":
            result = []
            setup = []
            collecting_setup = False
            for statement in node.body:
                # Keep checkpoint loading, device transfer, half and compile together.
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Tuple) and any(
                        isinstance(item, ast.Name) and item.id == "model" for item in target.elts)
                    for target in statement.targets):
                    collecting_setup = True
                if collecting_setup and assignment(statement, "detector"):
                    result.append(span("setup.wilor_model", setup))
                    collecting_setup = False
                if collecting_setup:
                    setup.append(statement)
                    continue
                if assignment(statement, "camera_summaries"):
                    result.append(call_statement("_ego_timer.attach(model)", statement))
                result.append(span("setup.detector", [statement]) if assignment(statement, "detector") else statement)
                if assignment(statement, "device"):
                    result.append(call_statement("_ego_timer.configure(device)", statement))
            assert not collecting_setup, "could not locate model setup boundary"
            node.body = result
            for statement in ast.walk(node):
                if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name) and statement.target.id == "camera":
                    statement.body.insert(0, call_statement("_ego_timer.camera = camera", statement))
            self.wrap_output_and_cleanup(node)
        return node

    def wrap_output_and_cleanup(self, node):
        class Wrap(ast.NodeTransformer):
            def visit_Expr(self, statement):
                if isinstance(statement.value, ast.Call):
                    function = ast.unparse(statement.value.func)
                    if function == "stream.write":
                        return span("output.jsonl", [statement])
                    if function in ("gc.collect", "torch.cuda.empty_cache"):
                        return span("cleanup", [statement])
                return statement
        Wrap().visit(node)


def instrument_source(source, filename):
    transform = Instrument()
    tree = transform.visit(ast.parse(source, filename=filename))
    required = {"infer_dual", "_prepare_dual_batched", "_infer_prepared_batched", "main", "load"}
    if not required.issubset(transform.functions):
        raise RuntimeError("WiLoR adapter structure changed; timing bootstrap needs updating")
    return compile(ast.fix_missing_locations(tree), filename, "exec")


def main():
    target = Path(sys.argv[1]).resolve()
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    code = instrument_source(target.read_text(encoding="utf-8"), str(target))
    namespace = {"__name__": "__main__", "__file__": str(target),
                 "__package__": None, "_ego_timer": timer, "_ego_timed": timed}
    exec(code, namespace)


if __name__ == "__main__":
    main()
