# Source work/group_003_wrist_outward_20261119/export_dataset.py, SHA-256 4edc43a24769c4ad4eecd2662a555bffdfb4ca10543fbc1c9b222151891d54d6

import case_settings as case

import numpy as np

import torch

from fit_sequence import EGO, import_mano, run_model

from native_camera_geometry import canonical_camera_rotation, rotate_mano_geometry, rotate_mano_pose

from mano_conventions import mirror_left_points

from wilor_label_projection import project_label_points

from export_multiview_wilor_training_dataset import _mesh_bbox

def reconstructed_poses(track):
    mano = import_mano(EGO/"third_party/MANO")
    model = mano.load(model_path=str(EGO/"models/mano"),is_rhand=True,use_pca=True,
                      num_pca_comps=45,flat_hand_mean=False,batch_size=1).eval()
    poses = []
    with torch.no_grad():
        for i in range(0,len(track["pair_indices"]),128):
            sl = slice(i,i+128)
            count = len(track["pair_indices"][sl])
            vertices,joints,output = run_model(model,torch.tensor(track["betas"]).expand(count,-1),
                *[torch.tensor(track[k][sl]) for k in ("global_orient","hand_pose_pca","translation")])
            np.testing.assert_allclose(vertices.numpy(),track["vertices"][sl],atol=2e-6,rtol=0)
            np.testing.assert_allclose(joints.numpy(),track["joints"][sl],atol=2e-6,rtol=0)
            poses.append(output.full_pose.numpy().reshape(count,16,3))
    return np.concatenate(poses)

def camera_label(track, idx, full_pose, transform, camera, side=1):
    r,t = transform
    canonical = canonical_camera_rotation(r,side)
    translation = track["translation"][idx]
    v,j,shift,_ = rotate_mano_geometry(track["vertices"][idx]-translation,track["joints"][idx]-translation,translation,canonical)
    shift += mirror_left_points(t,side)
    v,j,shift = [mirror_left_points(x,side).astype(np.float32) for x in (v,j,shift)]
    info = dict(image_space="native_raw",camera_model="KB",distortion=camera.distortion.astype(np.float64))
    K = camera.K.astype(np.float32)
    pixels = project_label_points(v+shift,dict(K=K,**info)).astype(np.float32)
    box = _mesh_bbox(pixels,camera.image_size,.1,.50)
    if box is None: raise ValueError("No visible mesh in a source-qualified view")
    ori,hand_pose = rotate_mano_pose(full_pose[idx],canonical)
    return dict(bbox=box,vertices=v,joints_3d=j,joints_2d=torch.from_numpy(pixels),side=np.asarray(side,np.float32),trans=shift,K=K,
        mano=dict(global_orient=ori,hand_pose=hand_pose,betas=track["betas"].astype(np.float32)),**info)
