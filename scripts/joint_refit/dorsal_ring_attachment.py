"""Place the second ring-finger marker dorsally, at its axial position from this hand's calibration."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/label_pipeline/calibration/personal_hand_calibration'))
from run import PROJECT, import_mano


def axial_position(point, joints):
    segments = np.diff(joints, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    directions = segments / lengths[:, None]
    distances = np.clip(((point - joints[:-1]) * directions).sum(1), 0., lengths)
    nearest = joints[:-1] + distances[:, None] * directions
    segment = np.linalg.norm(point - nearest, axis=1).argmin()
    return nearest[segment], directions[segment], float(segment + distances[segment] / lengths[segment])


def intersect_triangle(distances):
    weights = np.eye(3)
    section = list(weights[np.abs(distances) < 1e-12])
    for a, b in ((0, 1), (1, 2), (2, 0)):
        if distances[a] * distances[b] < 0:
            t = distances[a] / (distances[a] - distances[b])
            section.append((1 - t) * weights[a] + t * weights[b])
    return np.mean(section, axis=0) if section else None


def placement(model, aligned, calibration):
    # HandOffset is attachment 0. Four sites per finger follow in Thumb-to-Pinky order.
    row = 14
    if str(aligned['marker_names'][row + 3]) != 'FingerRing2':
        raise ValueError('Unexpected Ring2 marker ordering')
    with torch.no_grad():
        mesh = model(betas=torch.tensor(aligned['betas'])[None], global_orient=torch.zeros(1, 3),
                     hand_pose=-model.hand_mean[None], transl=torch.zeros(1, 3), return_tips=True)
    vertices, joints = mesh.vertices[0].numpy(), mesh.joints[0].numpy()
    faces = model.faces_tensor.numpy()
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    dorsal = np.cross(joints[4] - joints[0], joints[7] - joints[1])
    dorsal /= np.linalg.norm(dorsal)
    if dorsal[1] < 0:
        dorsal = -dorsal
    bones = model.lbs_weights.numpy()[faces].mean(axis=1).argmax(axis=1)
    initial_site = calibration['barycentric'][row] @ triangles[calibration['attachment_faces'][row]]
    foot, axis, along = axial_position(initial_site, joints[[10, 11, 12, 19]])
    distances = (triangles - foot) @ axis
    cosine = normals @ dorsal
    eligible = np.isin(bones, [10, 11]) & (cosine >= .85) & (distances.min(1) <= 0) & (distances.max(1) >= 0)
    best = None
    for face in np.flatnonzero(eligible):
        bary = intersect_triangle(distances[face])
        if bary is None:
            continue
        radial = bary @ triangles[face] - foot
        radial -= (radial @ axis) * axis
        score = (radial @ dorsal) / max(np.linalg.norm(radial), 1e-12)
        if best is None or score > best[0]:
            best = (score, face, bary)
    report = dict(calibration_chain_position=along, source_face=int(aligned['attachment_faces'][row]))
    if best is None:
        report['status'] = 'no_dorsal_face_at_calibrated_position'
        return report
    _, face, bary = best
    aligned['attachment_faces'][row] = face
    aligned['barycentric'][row] = bary
    report.update(status='placed', face=int(face), dorsal_cosine=float(cosine[face]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    model = import_mano(PROJECT / 'third_party/MANO').load(model_path=str(PROJECT / 'models/mano'),
        is_rhand=True, use_pca=True, num_pca_comps=45, flat_hand_mean=False, batch_size=1)
    hands = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for hand in ('left', 'right'):
        name = f'{hand}_pilot.npz'
        if not (args.source / name).is_file():
            continue
        with np.load(args.source / name) as z, np.load(args.calibration / name) as calibration:
            aligned = dict(z)
            hands[hand] = placement(model, aligned, calibration)
        if hands[hand]['status'] != 'placed':
            raise ValueError(f'{hand} Ring2: no dorsal attachment at the calibrated position')
        np.savez_compressed(args.output / name, **aligned)
    if not hands:
        raise ValueError('No aligned hands available for Ring2 placement')
    (args.output / 'placement.json').write_text(json.dumps(hands, indent=2) + '\n')


if __name__ == '__main__':
    main()
