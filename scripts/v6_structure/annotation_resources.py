"""Record verified public annotation options and existing-patient exclusions.

No downloads, registrations, external messages or expert judgements occur.
External cards are source-attributed, not project experimental results.
"""
import argparse
from pathlib import Path
import pandas as pd
from scripts.v4_convergence.core import digest,json_write
from scripts.v5_anatomy.run_pilot import project_path

ROOT=Path(__file__).resolve().parents[2]


def prepare(output):
    output=project_path(output)
    if not output.is_relative_to(ROOT/'reports'):raise ValueError('Use a new reports directory')
    rows=[];inputs={}
    for cohort,folder,splits in [
        ('generator_original',ROOT/'data/processed/ax_t1_128/splits',['train','val','test']),
        ('heldout_judge',ROOT/'data/processed/ax_t1_128_heldout/preflight_20261016_01/splits',['train','val'])]:
        for split in splits:
            path=folder/f'{split}.csv';frame=pd.read_csv(path)
            inputs[str(path.relative_to(ROOT))]=digest(path)
            for patient in sorted(frame.patient_id.unique()):
                rows.append(dict(patient_id=patient,mama_mia_duke_id='DUKE_'+patient.rsplit('_',1)[1],
                                 existing_cohort=cohort,existing_split=split,
                                 exclude_from_new_untouched_evaluation=True))
    resources=[
        dict(name='Duke breast/FGT/vessel expert segmentations',
             primary_url='https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/',
             methods_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10915139/',
             exact_source_readme='https://www.cancerimagingarchive.net/wp-content/uploads/duke_README.txt',
             label_type='manual tissue masks reviewed by breast radiologists',
             documented_sequence='fat-saturated gradient-echo T1 pre-contrast',
             exact_current_ax_t1_alignment_verified=False,
             use='candidate for a separately versioned, same-sequence anatomy experiment',
             access='TCIA supplemental data; check per-file format, geometry, licence and IDs before a scoped download'),
        dict(name='MAMA-MIA expert tumour segmentations',
             primary_url='https://github.com/LidiaGarrucho/MAMA-MIA',
             methods_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC11923173/',
             dataset_url='https://doi.org/10.7303/syn60868042',
             label_type='expert-corrected tumour/non-mass-enhancement segmentations',
             documented_sequence='DCE-MRI; experts use first post-contrast images',
             exact_current_ax_t1_alignment_verified=False,
             use='strong candidate for a separate lesion-aware DCE benchmark, not labels for existing ax-t1 arrays',
             access='Synapse account/token or Health-RI XNAT account; no account was created or download attempted'),
        dict(name='Breast-MRI-NACT-Pilot',
             primary_url='https://www.cancerimagingarchive.net/collection/breast-mri-nact-pilot/',
             label_type='enhancement/threshold-derived segmentations; do not equate all SEG objects with expert manual masks',
             documented_sequence='DCE-MRI',exact_current_ax_t1_alignment_verified=False,
             use='secondary candidate after inspecting annotation provenance and acquisition support')]
    routes=[
        dict(institution='AIIMS Bathinda, Department of Radiodiagnosis',
             official_url='https://aiimsbathinda.edu.in/Departments.aspx/Dept-Radiodiagnosis-11',
             public_department_email='radiodiagnosis@aiimsbathinda.in',
             purpose='ask whether a breast-MRI-qualified reviewer can support an academic study',
             contacted=False,availability_confirmed=False,breast_mri_expertise_of_assigned_reader_confirmed=False),
        dict(institution='PGIMER Chandigarh, Radio Diagnosis and Imaging',
             official_url='https://pgimer.edu.in/PGIMER_PORTAL/PGIMERPORTAL/Department/Global/JSP/home_dept.jsp?CP=Eng&deptid=d42',
             directory_url='https://pgimer.edu.in/PGIMER_PORTAL/PGIMERPORTAL/Department/Global/JSP/DeptContactNo.jsp',
             purpose='institutional research-review enquiry, ideally through the project supervisor',
             contacted=False,availability_confirmed=False,breast_mri_expertise_of_assigned_reader_confirmed=False)]
    output.mkdir(parents=True,exist_ok=False)
    json_write(output/'source_catalog.json',dict(checked_date='2026-11-16',resources=resources,reviewer_routes=routes,
        public_annotations_found=True,direct_labels_for_current_ax_t1_verified=False,
        reviewer_recruited=False,large_downloads_performed=False,external_messages_sent=False))
    pd.DataFrame(rows).to_csv(output/'excluded_patients.csv',index=False)
    json_write(output/'inputs.json',inputs)
    (output/'README.md').write_text(
        '# Annotation and reviewer routes\n\nProducer: `python -m scripts.v6_structure.annotation_resources`.\n'
        '- `source_catalog.json`: primary source URLs, annotation type, sequence compatibility and uncontacted institutional routes.\n'
        '- `excluded_patients.csv`: existing original/judge patients to exclude from any claimed new untouched Duke evaluation cohort.\n'
        '- `inputs.json`: hashes of local split files used for that exclusion list.\n\n'
        'Public expert annotations exist. They are not yet verified as aligned with the current non-fat-suppressed ax-t1 series.\n'
        'The earlier local audit found zero exact ax-t1 rows in the 2D tissue-mask sequence mapping. Same patient ID does not establish registration.\n'
        'Do not resize DCE masks onto current images. A separate DCE study would require its own protocol, splits and model/assessor validation.\n'
        'Published masks annotate real images, not newly generated outputs; independent output review or another validated assessment remains necessary.\n'
        'No reviewer has agreed to participate. Institutional contacts are enquiry routes, not confirmed breast-MRI experts or bookings.\n'
        'A bulk data download, account setup or outreach requires a concrete next action and appropriate user authorisation.\n',encoding='utf-8')
    (output/'reviewer_brief.md').write_text(
        '# Proposed academic review scope (not sent)\n\n'
        'The Thapar capstone studies counterfactual scanner-associated appearance editing of breast MRI.\n'
        'We seek a qualified breast-MRI reviewer, introduced through the academic supervisor where possible.\n'
        'The initial task is blinded comparison of source/output slices: body/breast outline, chest-wall location,\n'
        'internal-structure correspondence, and whether lesion preservation is assessable. Mark unassessable cases explicitly.\n'
        'Processed outputs are 128x128; they may be inadequate for lesion assessment and do not constitute clinical images.\n'
        'Anatomy and lesion judgements must stay separate from scanner target hits. No desired positive result is implied.\n'
        'Case count, compensation (if any), institutional requirements, qualification and reader study design remain to be agreed.\n'
        'The existing shuffled review pack has image pairs, blank responses and native-grid mask/landmark templates.\n'
        'No patient material or request has been sent to an institution.\n',encoding='utf-8')
    print(output,flush=True);return output


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);print(prepare(p.parse_args().output))
