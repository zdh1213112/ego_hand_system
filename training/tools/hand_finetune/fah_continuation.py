"""Continue a versioned FAH run at a validated optimizer step."""
import json
import random

import numpy as np
import torch

from .fah_experiments import digest, snapshot


class EpochSampler(torch.utils.data.Sampler):
    def __init__(self, size, seed):
        self.size, self.seed = size, seed
        self.epoch, self.offset = 1, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch - 1)
        return iter(torch.randperm(self.size, generator=generator).tolist()[self.offset:])

    def __len__(self):
        return self.size - self.offset


def save_training_state(out, model, optimizer, report):
    params = {k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad}
    state = dict(step=report["latest"]["step"], checkpoint_sha256=report["latest"]["checkpoint_sha256"],
                 model=params, optimizer=optimizer.state_dict(), python_rng=random.getstate(),
                 numpy_rng=np.random.get_state(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    tmp = out / "training_state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(out / "training_state.pt")


def continue_run(source, out, model, optimizer, report):
    prior = json.loads((source / "metrics.json").read_text())
    previous, current = prior["protocol"], report["protocol"]
    mutable = {"steps", "eval_every", "deployment_sha256", "validation_schedule"}
    if ({k: v for k, v in previous.items() if k not in mutable}
            != {k: v for k, v in current.items() if k not in mutable}):
        raise ValueError("Continuation data, sampling, initialization or training protocol differs")
    for key in ("version", "components", "component_settings", "optimizer_groups"):
        if prior[key] != report[key]:
            raise ValueError(f"Continuation differs in {key}")
    latest, best = prior["latest"], prior["best"]
    if not (best and 0 < latest["step"] < current["steps"]):
        raise ValueError("Continuation needs a validated training step below the new total budget")
    state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
    if (state["step"] != latest["step"] or state["checkpoint_sha256"] != latest["checkpoint_sha256"]
            or digest(source / "model.pt") != latest["checkpoint_sha256"]
            or digest(source / "best_model.pt") != best["checkpoint_sha256"]):
        raise ValueError("Saved training state and validation checkpoints disagree")
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state["model"].keys():
        raise ValueError("Trainable parameters differ from the saved state")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state["model"][name])
    optimizer.load_state_dict(state["optimizer"])
    for key in ("baseline", "epochs", "release_validation_report", "release_mesh_cohorts", "best", "latest"):
        report[key] = prior[key]
    for entry in [prior["release_validation_report"], *(e["native_validation"]["validation_report"] for e in prior["epochs"])]:
        snapshot(source / entry, out / entry)
    snapshot(source / "best_model.pt", out / "best_model.pt")
    snapshot(source / "model.pt", out / "model.pt")
    current["deployment_sha256"] = previous["deployment_sha256"]
    schedule = current["validation_schedule"]
    current["validation_schedule"] = [s for s in previous["validation_schedule"] if s["after_step"] < latest["step"]]
    if current["validation_schedule"][-1]["interval"] != schedule[0]["interval"]:
        current["validation_schedule"].append(dict(after_step=latest["step"], interval=schedule[0]["interval"]))
    report["continued_from"] = dict(run=str(source.resolve()), step=latest["step"], checkpoint_sha256=latest["checkpoint_sha256"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return latest["step"]
