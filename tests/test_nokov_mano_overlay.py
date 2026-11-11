from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from render_nokov_mano_overlay import association,draw_overlay


class NokovManoOverlayTests(unittest.TestCase):
    def test_offset_applied_once_and_no_endpoint_extrapolation(self):
        origin=1_794_000_000_000_000_000
        samples=np.array([origin,origin+11_000_000,origin+22_000_000],dtype=np.int64)
        delta=12_000_000
        queries=np.array([origin+delta-1,origin+delta,origin+delta+6_000_000,
                          origin+delta+22_000_000,origin+delta+22_000_001],dtype=np.int64)
        self.assertEqual(association(queries,samples,delta),[None,0,1,2,None])

    def test_masked_points_and_behind_camera_triangles_do_not_draw(self):
        image=np.full((60,80,3),100,dtype=np.uint8)
        uv=np.full((823,2),np.nan)
        optical=np.zeros((823,3));optical[:,2]=-1
        inside=np.zeros(823,dtype=bool)
        result=draw_overlay(image.copy(),(uv,inside,optical),np.array([[0,1,2]]))
        np.testing.assert_array_equal(result,image)


if __name__=='__main__':unittest.main()
