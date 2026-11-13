from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_native_multiview_observations import unique_hand_matches,geometry_filter
from ego_data.calibration import CameraCalibration
from camera_models import project_points


def observations(point):
    result=[]
    for i,(x,y) in enumerate([(0,0),(.1,0),(0,.1),(-.1,0)]):
        T=np.eye(4);T[:2,3]=[x,y]
        c=CameraCalibration(f'camera{i}','camera','KB',(1920,1200),np.array([[500,0,960],[0,500,600],[0,0,1.]]),np.zeros(4),T)
        pixel,_=project_points(c,(np.asarray(point)-T[:3,3])[None])
        result.append((c.camera_id,c,pixel[0]))
    return result


def test_rgb_consensus_rejects_bad_view_without_nokov_gate():
    obs=observations([.02,.03,.6]);bad=list(obs);name,c,p=bad[3];bad[3]=(name,c,p+[160,120])
    result,reason=geometry_filter(bad)
    assert reason=='accepted'
    point,error,inlier,condition=result
    np.testing.assert_allclose(point,[.02,.03,.6],atol=1e-6)
    assert inlier.tolist()==[True,True,True,False]


def test_weak_or_two_view_geometry_is_not_promoted():
    assert geometry_filter(observations([0,0,.6])[:2])[1]=='insufficient_views'
    result,reason=geometry_filter(observations([0,0,10]))
    assert result is None and reason=='weak_geometry'


def test_cross_hand_blob_ambiguity_is_dropped():
    a=(0,1,np.array([100.,100.]),1.);b=(1,8,np.array([101.,100.]),1.5)
    assert unique_hand_matches([a,b])==[]
    b=(1,8,np.array([101.,100.]),6.)
    result=unique_hand_matches([a,b]);assert len(result)==1 and result[0][0]==0
