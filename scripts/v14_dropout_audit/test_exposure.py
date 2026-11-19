from collections import Counter
import unittest
import pandas as pd

from scripts.v14_dropout_audit.exposure import count_record


class ExposureChecks(unittest.TestCase):
    def test_distinct_zero_absent_and_label_drop_counts(self):
        source=pd.DataFrame(dict(patient_id=['a','b','c'],manufacturer_label=[0,1,0],field_strength_label=[0,1,1]))
        row=dict(training_seed=1,mode='fine',epoch_at_draw=0,sample_indices=[0,1,2],timesteps=[100,500,100],
                 condition_dropped=[False,True,False],gate_states=[0,3,4],patient_ids=['a','b','c'])
        joint,epochs,patients=Counter(),Counter(),Counter()
        count_record(row,source,joint,epochs,patients)
        self.assertEqual(sum(joint.values()),3)
        self.assertEqual(joint[(1,'fine',0,False,True,0,0)],1)
        self.assertEqual(joint[(1,'fine',3,True,False,1,1)],1)
        self.assertEqual(joint[(1,'fine',4,False,True,0,1)],1)
        row['patient_ids'][0]='wrong'
        with self.assertRaises(ValueError):count_record(row,source,Counter(),Counter(),Counter())


if __name__=='__main__':unittest.main()
