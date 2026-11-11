"""Opt-in pipeline/component timing; no torch import or hooks when disabled."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
import csv
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time


def statistics(values):
    values = sorted(values)
    if not values:
        return {"calls": 0}

    def percentile(q):
        index = (len(values) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "calls": len(values), "total_ms": sum(values),
        "mean_ms": sum(values) / len(values), "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95), "min_ms": values[0], "max_ms": values[-1],
    }


class Timer:
    def __init__(self, directory=None, warmup_batches=1):
        self.directory = Path(directory) if directory else None
        self.warmup_batches = warmup_batches
        self.device = None
        self.torch = None
        self.camera = None
        self.batch_index = -1
        self.batch_size = None
        self.rows = []
        self.pending = []
        self.handles = []
        self.notes = []
        self.lock = threading.Lock()
        if self.directory:
            atexit.register(self.write)

    @property
    def enabled(self):
        return self.directory is not None

    def configure(self, device):
        if self.enabled:
            import torch
            self.device = device
            self.torch = torch

    def start(self, name, gpu=False, model_component=False):
        if not self.enabled:
            return None
        row = {
            "name": name, "camera": self.camera,
            "batch_index": self.batch_index if model_component else None,
            "hypotheses": self.batch_size if model_component else None,
            "warmup": model_component and self.batch_index < self.warmup_batches,
            "status": "ok", "cuda_ms": None,
        }
        events = None
        if gpu and self.device is not None and self.device.type == "cuda":
            # Record on the selected device's current stream, including cuda:N.
            with self.torch.cuda.device(self.device):
                events = (self.torch.cuda.Event(enable_timing=True),
                          self.torch.cuda.Event(enable_timing=True))
                events[0].record()
        return row, time.perf_counter(), events

    def stop(self, token, status="ok", flush=False):
        if token is None:
            return
        row, started, events = token
        if events:
            with self.torch.cuda.device(self.device):
                events[1].record()
        row["host_wall_ms"] = (time.perf_counter() - started) * 1000
        row["elapsed_ms"] = row["host_wall_ms"]
        row["status"] = status
        with self.lock:
            self.rows.append(row)
            if events:
                self.pending.append((row, events))
        if flush:
            self.flush()
            row["elapsed_ms"] = (time.perf_counter() - started) * 1000

    def flush(self):
        with self.lock:
            pending, self.pending = self.pending, []
        if pending:
            self.torch.cuda.synchronize(self.device)
            for row, (start, end) in pending:
                row["cuda_ms"] = start.elapsed_time(end)

    @contextmanager
    def span(self, name, gpu=False):
        token = self.start(name, gpu)
        try:
            yield
        except BaseException:
            self.stop(token, status="error", flush=gpu)
            raise
        else:
            self.stop(token, flush=gpu)

    def iterate(self, name, iterable):
        iterator = iter(iterable)
        while True:
            token = self.start(name)
            try:
                value = next(iterator)
            except StopIteration:
                return
            except BaseException:
                self.stop(token, status="error")
                raise
            self.stop(token)
            yield value

    def attach(self, model):
        """Hook coarse components, avoiding per-layer instrumentation overhead."""
        if not self.enabled:
            return
        active = {}

        def begin(name):
            active.setdefault(name, []).append(self.start(name, True, True))

        def finish(name, output):
            if not active.get(name):
                return  # An exception may occur before a boundary's start hook.
            token = active[name].pop()
            self.stop(token, status="error" if output is None else "ok")

        def hook(module, name):
            self.handles.append(module.register_forward_pre_hook(
                lambda m, inputs: begin(name)))
            self.handles.append(module.register_forward_hook(
                lambda m, inputs, output: finish(name, output), always_call=True))

        def model_start(module, inputs):
            self.batch_index += 1
            self.batch_size = int(inputs[0]["img"].shape[0])
            self.mano_call = 0
            begin("model.total")

        def model_end(module, inputs, output):
            if output is None:
                for name, stack in active.items():
                    if name != "model.total":
                        while stack:
                            self.stop(stack.pop(), status="error")
            total_token = active["model.total"][-1]
            finish("model.total", output)
            self.flush()  # Once per forward, not once per layer.
            total_token[0]["elapsed_ms"] = (time.perf_counter() - total_token[1]) * 1000

        self.handles.append(model.register_forward_pre_hook(model_start))
        self.handles.append(model.register_forward_hook(model_end, always_call=True))
        hook(model.backbone, "backbone.total")
        backbone = model.backbone
        if hasattr(backbone, "_orig_mod"):
            self.notes.append(
                "Compiled backbone: internal ViT/decoder hooks omitted to avoid "
                "graph breaks. Use --compile-backbone 0 for full breakdown."
            )
        else:
            # This WiLoR embeds its MANO parameter head in ViT.forward_features.
            self.handles.append(backbone.register_forward_pre_hook(
                lambda m, inputs: begin("vit.features")))
            self.handles.append(backbone.last_norm.register_forward_hook(
                lambda m, inputs, output: finish("vit.features", output),
                always_call=True))
            self.handles.append(backbone.decpose.register_forward_pre_hook(
                lambda m, inputs: begin("mano.parameter_decoder")))
            self.handles.append(backbone.register_forward_hook(
                lambda m, inputs, output: finish("mano.parameter_decoder", output),
                always_call=True))
            hook(backbone.patch_embed, "vit.patch_embed")
            self.handles.append(backbone.blocks[0].register_forward_pre_hook(
                lambda m, inputs: begin("vit.transformer_blocks")))
            self.handles.append(backbone.blocks[-1].register_forward_hook(
                lambda m, inputs, output: finish("vit.transformer_blocks", output),
                always_call=True))
            for name in ("decpose", "decshape", "deccam"):
                hook(getattr(backbone, name), "mano.decoder." + name)

        def mano_start(module, inputs):
            self.mano_call += 1
            begin("mano.coarse" if self.mano_call == 1 else "mano.final")

        def mano_end(module, inputs, output):
            finish("mano.coarse" if self.mano_call == 1 else "mano.final", output)

        self.handles.append(model.mano.register_forward_pre_hook(mano_start))
        self.handles.append(model.mano.register_forward_hook(mano_end, always_call=True))
        hook(model.refine_net, "refinenet.total")
        for name in ("deconv", "dec_pose", "dec_shape", "dec_cam"):
            hook(getattr(model.refine_net, name), "refinenet." + name)

    def write(self):
        if not self.enabled:
            return
        self.flush()
        self.directory.mkdir(parents=True, exist_ok=True)
        components = {}
        for name in sorted({row["name"] for row in self.rows}):
            rows = [row for row in self.rows if row["name"] == name]
            entry = {"error_calls": sum(row["status"] != "ok" for row in rows)}
            successful = [row for row in rows if row["status"] == "ok"]
            for metric in ("host_wall_ms", "cuda_ms", "elapsed_ms"):
                measured = [row for row in successful if row[metric] is not None]
                entry[metric] = {
                    "all": statistics([row[metric] for row in measured]),
                    "steady": statistics([row[metric] for row in measured if not row["warmup"]]),
                    "first_ms": measured[0][metric] if measured else None,
                }
            components[name] = entry
        report = {
            "schema_version": 1, "device": str(self.device),
            "warmup_batches": self.warmup_batches,
            "warmup_policy": "first N model forward attempts across all cameras",
            "notes": self.notes + [
                "Times are inclusive; parent/child and overlapped decode times must not be added.",
                "host_wall_ms is host elapsed time; CUDA hooks measure stream event intervals, not kernel-only time.",
                "CUDA synchronization once per forward and detector/transfer span changes pipeline overlap.",
                "First calls include lazy initialization/compilation; steady excludes warmup only for model components.",
                "hypotheses counts hand crops (two handedness hypotheses per detection), not source frames.",
            ],
            "components": components,
        }
        (self.directory / "components.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (self.directory / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["name", "camera", "batch_index", "hypotheses", "warmup",
                      "status", "host_wall_ms", "cuda_ms", "elapsed_ms"]
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(self.rows)


timer = Timer(os.environ.get("EGO_WILOR_TIMING_DIR"),
              int(os.environ.get("EGO_WILOR_TIMING_WARMUP", "1")))


def timed(name, gpu=False):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not timer.enabled:
                return function(*args, **kwargs)
            with timer.span(name, gpu):
                return function(*args, **kwargs)
        return wrapped
    return decorate
