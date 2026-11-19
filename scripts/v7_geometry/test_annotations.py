"""DICOM SEG source identity and geometry rejection checks."""
import unittest
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from scripts.v7_geometry.annotation_compatibility import check_source
from scripts.v7_geometry.native_annotations import frame_reference,frame_geometry


def fixture():
    source=Dataset();source.SOPInstanceUID='2.25.11';source.SeriesInstanceUID='2.25.12'
    source.Rows=8;source.Columns=9;source.ImageOrientationPatient=[1,0,0,0,1,0]
    source.ImagePositionPatient=[1,2,3];source.PixelSpacing=[.6,.7]
    seg=Dataset();seg.Rows=8;seg.Columns=9
    series=Dataset();series.SeriesInstanceUID=source.SeriesInstanceUID
    ref=Dataset();ref.ReferencedSOPInstanceUID=source.SOPInstanceUID
    series.ReferencedInstanceSequence=Sequence([ref]);seg.ReferencedSeriesSequence=Sequence([series])
    shared=Dataset();orientation=Dataset();orientation.ImageOrientationPatient=source.ImageOrientationPatient
    measures=Dataset();measures.PixelSpacing=source.PixelSpacing
    shared.PlaneOrientationSequence=Sequence([orientation]);shared.PixelMeasuresSequence=Sequence([measures])
    seg.SharedFunctionalGroupsSequence=Sequence([shared])
    frame=Dataset();position=Dataset();position.ImagePositionPatient=source.ImagePositionPatient
    frame.PlanePositionSequence=Sequence([position]);derivation=Dataset()
    derivation.SourceImageSequence=Sequence([ref]);frame.DerivationImageSequence=Sequence([derivation])
    return seg,frame,source


class AnnotationTests(unittest.TestCase):
    def test_matching_reference_is_not_automatic_ground_truth_approval(self):
        seg,frame,source=fixture();result=check_source(seg,source)
        self.assertTrue(result['source_sop_referenced']);self.assertFalse(result['direct_transfer_approved'])

    def test_same_grid_but_wrong_sop_or_series_is_rejected(self):
        seg,frame,source=fixture();source.SOPInstanceUID='2.25.99'
        self.assertFalse(check_source(seg,source)['source_sop_referenced'])
        source.SeriesInstanceUID='2.25.100'
        self.assertEqual(check_source(seg,source)['status'],'different_or_unresolved_series')

    def test_single_frame_reference_required(self):
        seg,frame,source=fixture();self.assertEqual(frame_reference(frame),'2.25.11')
        another=Dataset();another.ReferencedSOPInstanceUID='2.25.98'
        frame.DerivationImageSequence[0].SourceImageSequence.append(another)
        with self.assertRaises(ValueError):frame_reference(frame)

    def test_native_geometry_exact_match(self):
        seg,frame,source=fixture();result=frame_geometry(seg,frame,source)
        self.assertEqual(result['position_error_mm'],0);self.assertEqual(result['spacing_error_mm'],0)

    def test_shifted_plane_and_wrong_spacing_rejected(self):
        seg,frame,source=fixture();frame.PlanePositionSequence[0].ImagePositionPatient=[1,2,4]
        with self.assertRaises(ValueError):frame_geometry(seg,frame,source)
        seg,frame,source=fixture();source.PixelSpacing=[1.,1.]
        with self.assertRaises(ValueError):frame_geometry(seg,frame,source)

    def test_mismatched_native_dimensions_and_missing_geometry_rejected(self):
        seg,frame,source=fixture();source.Rows=7
        with self.assertRaises(ValueError):frame_geometry(seg,frame,source)
        seg,frame,source=fixture();del frame.PlanePositionSequence
        with self.assertRaises(ValueError):frame_geometry(seg,frame,source)


if __name__=='__main__':unittest.main(verbosity=2)
