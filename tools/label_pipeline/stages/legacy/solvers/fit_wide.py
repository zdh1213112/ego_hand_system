# Source work/direction_acceptance_20261119/fit_wide.py, SHA-256 825ff36d24b0b84f02486d18536d5fa0520ae172f8d19f1912ae41afce4f0879

import case_settings as case

import numpy as np

ROOT = case.stage('direction_acceptance_20261119')

WORK = ROOT.parent

from rounds import Fit, HANDS, WEIGHTS, run

def folder(group):
    return WORK / f"{group}_anatomy_20261119"

def source_track(group, hand):
    return folder(group) / "wide3" / (hand + ".npz")

def build(group, hand):
    dest = folder(group)
    dest.mkdir(exist_ok=True)
    fit = Fit(group, hand, bounds_folder=dest / "bounds")
    c = dest / "C" / (hand + ".npz")
    weight = WEIGHTS[hand]
    if not c.exists():
        ii = np.arange(len(fit.ts))
        strict, _ = fit.solve(ii, weight)
        q, ii, relaxed = fit.run("relaxed", weight, ii, prior_q=strict)
        fit.store("C", q, ii, relaxed, weight, destination=c)
    run(group, hand, "wide", folder=dest, seed_path=c, fit=fit)
