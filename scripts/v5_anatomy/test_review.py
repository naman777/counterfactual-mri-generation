"""Review ingestion tests: absent labels, attribution and grid compatibility."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.v5_anatomy.score_review import score


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.pilot=self.root/'pilot';(self.pilot/'review').mkdir(parents=True);(self.pilot/'arrays').mkdir()
        (self.pilot/'complete.json').write_text(json.dumps({'complete':True}))
        pd.DataFrame([dict(review_id='a',case_id='field_strength_000',variant='sdedit')]).to_csv(
            self.pilot/'review/private_mapping.csv',index=False)
        pd.DataFrame([dict(review_id='a',judge='heldout0',axis='field_strength',variant='sdedit',target_hit=True)]).to_csv(
            self.pilot/'predictions.csv',index=False)
        np.savez(self.pilot/'arrays/field_strength_000.npz',source=np.zeros((128,128),np.float32))

    def tearDown(self):
        self.temp.cleanup()

    def review(self,**kwargs):
        path=self.root/'review.csv'
        pd.DataFrame([dict(review_id='a',reviewer='',anatomy_preserved='',lesion_assessable='',lesion_preserved='')|kwargs]).to_csv(path,index=False)
        return path

    def annotations(self,**kwargs):
        path=self.root/'annotations.json'
        path.write_text(json.dumps({'cases':[dict(review_id='a',reviewer=None,source_mask=None,edit_mask=None,
                                                 source_landmarks=None,edit_landmarks=None)|kwargs]}))
        return path

    def test_blank_template_has_zero_coverage(self):
        output=score(self.pilot,self.root/'out',self.review(),self.annotations())
        coverage=json.loads((output/'coverage.json').read_text())
        self.assertEqual(coverage['cases_with_anatomy_judgement'],0)
        self.assertFalse(coverage['missing_judgements_replaced'])
        for name in ['reader_responses.csv','annotation_metrics.csv','joint_reader_judge.csv']:
            self.assertTrue(pd.read_csv(output/name).empty)

    def test_positive_answer_without_reviewer_rejected(self):
        with self.assertRaises(ValueError):
            score(self.pilot,self.root/'out',self.review(anatomy_preserved='yes'))
        self.assertFalse((self.root/'out').exists())

    def test_unassessed_lesion_does_not_become_joint_preservation(self):
        output=score(self.pilot,self.root/'out',self.review(reviewer='reader1',anatomy_preserved='yes'))
        result=pd.read_csv(output/'joint_reader_judge.csv')
        self.assertTrue(pd.isna(result.iloc[0].target_hit_and_reviewed_preservation))

    def test_explicit_reader_failure_stays_failure_despite_hit(self):
        output=score(self.pilot,self.root/'out',self.review(reviewer='reader1',anatomy_preserved='no',lesion_assessable='no'))
        result=pd.read_csv(output/'joint_reader_judge.csv')
        self.assertFalse(result.iloc[0].target_hit_and_reviewed_preservation)

    def test_foreign_grid_rejected_without_resizing(self):
        np.save(self.root/'mask.npy',np.ones((64,64),np.uint8))
        with self.assertRaises(ValueError):
            score(self.pilot,self.root/'out',annotations=self.annotations(
                reviewer='reader1',source_mask='mask.npy',edit_mask='mask.npy'))

    def test_masks_and_known_landmark_displacement_scored(self):
        mask=np.zeros((128,128),np.uint8);mask[30:90,40:90]=1
        np.save(self.root/'mask.npy',mask)
        output=score(self.pilot,self.root/'out',annotations=self.annotations(reviewer='reader1',
            source_mask='mask.npy',edit_mask='mask.npy',source_landmarks={'a':[10,20]},edit_landmarks={'a':[13,24]}))
        result=pd.read_csv(output/'annotation_metrics.csv').iloc[0]
        self.assertEqual(result.annotated_dice,1.)
        self.assertEqual(result.landmark_mean_shift_px,5.)


if __name__=='__main__':
    unittest.main(verbosity=2)
