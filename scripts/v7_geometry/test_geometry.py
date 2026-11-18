"""Physical-coordinate and exact pixel/label/landmark round-trip checks."""
import unittest
import numpy as np
import pandas as pd
from scripts.v7_geometry.orientation import orientation_plan,apply_orientation,invert_orientation,resized_pixel_map,transform_landmarks
from scripts.v7_geometry.support import request_support,require_research_support


class GeometryTests(unittest.TestCase):
    def test_eight_signed_permutations_preserve_pixels_exactly(self):
        image=np.arange(35,dtype=np.float32).reshape(5,7)
        for transpose in [False,True]:
            for a in [-1,1]:
                for b in [-1,1]:
                    row=np.array([a,0,0]) if transpose else np.array([0,a,0])
                    col=np.array([0,b,0]) if transpose else np.array([b,0,0])
                    plan=orientation_plan(np.r_[col,row],image.shape)
                    canonical=apply_orientation(image,plan)
                    np.testing.assert_array_equal(invert_orientation(canonical,plan),image)
                    transform=np.asarray(plan['canonical_to_source_index'])
                    for r in range(canonical.shape[0]):
                        for c in range(canonical.shape[1]):
                            old=transform@np.array([r,c,1]);self.assertEqual(canonical[r,c],image[int(old[0]),int(old[1])])

    def test_ge_like_flip_and_positive_lps_frame(self):
        plan=orientation_plan([-1,0,0,0,-1,0],(2,3))
        self.assertTrue(plan['flip_rows'] and plan['flip_columns'])
        physical=resized_pixel_map([-1,0,0,0,-1,0],[10,20,30],[2,3],(2,3),(2,3),plan)
        canonical=np.asarray(physical['canonical_pixel_to_lps'])
        np.testing.assert_allclose(canonical[:,2],[4,18,30])
        np.testing.assert_allclose(canonical[:,0],[0,2,0]);np.testing.assert_allclose(canonical[:,1],[3,0,0])

    def test_first_iop_triplet_moves_columns_not_rows(self):
        plan=orientation_plan([1,0,0,0,1,0],(3,4))
        result=np.asarray(resized_pixel_map([1,0,0,0,1,0],[10,20,30],[2,3],(3,4),(3,4),plan)['source_pixel_to_lps'])
        np.testing.assert_allclose(result@np.array([1,0,1]),[10,22,30])
        np.testing.assert_allclose(result@np.array([0,1,1]),[13,20,30])

    def test_resize_centres_and_transposed_physical_mapping(self):
        iop=[0,-1,0,1,0,0];plan=orientation_plan(iop,(3,4))
        physical=resized_pixel_map(iop,[5,9,12],[2,3],(6,8),(3,4),plan)
        a=np.asarray(physical['source_pixel_to_lps']);b=np.asarray(physical['canonical_pixel_to_lps'])
        for point in [[0,0,1],[1,1,1],[3,2,1]]:
            np.testing.assert_allclose(b@point,a@np.asarray(plan['canonical_to_source_index'])@point)
        np.testing.assert_allclose(a[:,2],[6,7.5,12])

    def test_label_masks_and_landmarks_roundtrip_without_interpolation(self):
        mask=np.zeros((5,7),np.uint8);mask[1:4,2:6]=1
        plan=orientation_plan([0,-1,0,-1,0,0],mask.shape)
        np.testing.assert_array_equal(invert_orientation(apply_orientation(mask,plan),plan),mask)
        points={'a':[1,2],'b':[6,4]}
        restored=transform_landmarks(transform_landmarks(points,plan),plan,to_source=True)
        for key in points:np.testing.assert_allclose(restored[key],points[key])

    def test_missing_nonaxial_oblique_and_nonorthonormal_rejected(self):
        angle=np.deg2rad(10)
        for iop in [None,[1,0,0,0,0,1],[2,0,0,0,1,0],
                    [np.cos(angle),np.sin(angle),0,-np.sin(angle),np.cos(angle),0]]:
            with self.assertRaises(ValueError):orientation_plan(iop)

    def test_wrong_grid_and_invalid_spacing_rejected(self):
        plan=orientation_plan([1,0,0,0,1,0],(5,7))
        with self.assertRaises(ValueError):apply_orientation(np.zeros((7,5)),plan)
        with self.assertRaises(ValueError):resized_pixel_map([1,0,0,0,1,0],[0,0,0],[0,1],(5,7),(5,7),plan)

    def test_condition_support_does_not_become_causal_approval(self):
        train=pd.DataFrame(dict(manufacturer_label=[0]*6,field_strength_label=[1]*6,
                                echo_time_norm=[.3,.4,.5,.6,.7,.8],patient_id=list('abcdef')))
        observed=request_support(train,0,1,.4)
        self.assertEqual(observed['condition_support_status'],'observed_condition')
        self.assertFalse(observed['clinical_approval']);self.assertFalse(observed['causal_edit_identified'])
        self.assertEqual(request_support(train,0,1,.45)['condition_support_status'],'interpolation_only')
        outside=request_support(train,0,1,.1)
        self.assertEqual(outside['condition_support_status'],'outside_target_echo_range')
        with self.assertRaises(ValueError):require_research_support(outside)
        self.assertEqual(require_research_support(outside,True),'research_only_not_clinically_validated')
        self.assertEqual(request_support(train,1,1,.4)['condition_support_status'],'missing_target_cell')
        self.assertEqual(request_support(train.iloc[:2],0,1,.4)['condition_support_status'],'sparse_target_cell')


if __name__=='__main__':unittest.main(verbosity=2)
