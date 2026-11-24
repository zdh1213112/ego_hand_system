# Source work/mcp_outward_20261119/constraints.py, SHA-256 f4e6c71c8cca4586b68627946373aa1ad7f098fe713dad98abd1db0ab9706518

import case_settings as case

import torch

import torch.nn.functional as F

from fit_outward import ROOTS

def palm(j):
    n = F.normalize(torch.cross(j[:, 5] - j[:, 1], j[:, 17] - j[:, 1], dim=-1), dim=-1)
    b = F.normalize(j[:, ROOTS + 1] - j[:, ROOTS], dim=-1)
    return torch.rad2deg(torch.asin((b * n[:, None]).sum(-1).clamp(-0.99999, 0.99999)))
