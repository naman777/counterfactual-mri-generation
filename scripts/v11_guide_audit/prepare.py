"""Record the exploratory analysis before inspecting detailed control outcomes."""
import argparse
import json
from scripts.v8_generator_dev.io import ROOT, digest, json_write, project_path, verify_inputs


def prepare(run_id):
    parent = ROOT / 'reports/v10_detail'
    verified = parent / 'final_20261117_02_verified'
    metadata = parent / 'metadata_verification_20261117_01'
    verify_inputs(json.loads((verified / 'inputs.json').read_text()))
    verify_inputs(json.loads((metadata / 'inputs.json').read_text()))
    assert json.loads((verified / 'complete.json').read_text())['saved_outputs_verified']
    receipt = parent / 'results_20261117_01/complete.json'
    p = json.loads((parent / 'preparation_20261117_02/protocol.json').read_text())
    protocol = dict(format='exploratory_guide_control_diagnosis_v1',
        previous_results=receipt.relative_to(ROOT).as_posix(),
        previous_protocol=(parent / 'preparation_20261117_02/protocol.json').relative_to(ROOT).as_posix(),
        training_seeds=p['training_seeds'], generation_seeds=[p['evaluation']['selection_generation_seed'],
            *p['evaluation']['confirmation_generation_seeds']],
        patients=15, bootstrap_seed=583361, bootstrap_resamples=10000,
        intervals='Equal-patient crossed bootstrap over patients and training seeds; generation seeds averaged in each cell; paired patient intervals within each training seed',
        axes={'manufacturer':'primary diagnostic axis', 'field_strength':'exploratory correct-guide mode contrasts only'},
        contrasts=['fine_minus_coarse at every available guide control',
            'correct_minus_each_other_control within each guided mode',
            'interaction: (fine_minus_coarse under correct) minus (fine_minus_coarse under each other control)',
            'guided_correct_minus_learned_zero_correct', 'learned_zero_correct_minus_absent'],
        metrics=['target_hit','null_hit','paired_hit_delta','target_probability','null_probability','probability_shift',
            'other_axis_stable','interior_mad','null_interior_mad','interior_hf_residual_mad',
            'null_interior_hf_residual_mad','ssim','null_ssim','null_to_target_interior_mad','null_to_target_hf_mad'],
        strata=['all','each edit direction','each support status','each source-warning status'],
        interactions_family=dict(axis='manufacturer',metric='paired_hit_delta',contrasts=5,judges=3,
            family_size=15,alpha=.05,quantiles=[.05/(2*15),1-.05/(2*15)]),
        uncertainty_warning='Other intervals are descriptive 95% and not adjusted for the many exploratory comparisons. No significance-based profile selection or new advancement decision.',
        case_audit='Every patient and training seed: decompose primary editing differences into target and null hit changes; preserve unchanged, gain, loss and mixed generation outcomes. No selected-case-only numerical summary.',
        new_array_diagnostics='Absolute target-minus-null change and its Gaussian sigma1 truncate3 high-pass within the original source mask eroded8; all invalid regions retained as unavailable.',
        interpretation=['The primary outcome and completed advancement failure are already known; this is post-hoc reused-development diagnosis.',
            'Wrong/zero/split guides were not trained as interventions; distribution shift can explain degradation. Controls do not prove copying, scanner invariance or clinical preservation.',
            'Fine-only in the coarse model is the second duplicated coarse block; it is not fine information.',
            'No new sampler, training, model selection, test pixels, paid app, download or scientific gate changes.'],
        panel_rule='All 15 patients, all 3 training seeds at the already fixed selection generation seed; primary source/coarse/fine/null/difference panels and a full correct/wrong/zero/absent comparison for each seed.')
    output = ROOT / 'reports/v11_guide_audit' / run_id
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / 'protocol.json', protocol)
    paths = [receipt, project_path(protocol['previous_protocol']), verified / 'complete.json', verified / 'inputs.json',
        metadata / 'complete.json', metadata / 'inputs.json', project_path(__file__),
        ROOT / 'scripts/v10_detail/engine.py', ROOT / 'scripts/v6_structure/model.py']
    for relative in json.loads(receipt.read_text())['evaluations'].values():
        paths.extend(project_path(relative) / n for n in ('per_image.csv','requests.csv','protocol.json','complete.json'))
    json_write(output / 'inputs.json', {path.relative_to(ROOT).as_posix(): digest(path) for path in paths})
    (output / 'README.md').write_text('Producer: scripts.v11_guide_audit.prepare. protocol.json: exploratory contrasts, '
        'bootstrap/multiplicity/stratum/panel rules and limits fixed after the primary outcome was known, before detailed '
        'control analysis; inputs.json: prior independent verification, complete reports and implementation identity; '
        'complete.json: local-only preparation scope. No new efficacy or paid permission.\n', encoding='utf-8')
    json_write(output / 'complete.json', dict(complete=True, exploratory=True, primary_outcome_already_known=True,
        paid_gpu_calls=0, default_promoted=False, clinical_preservation='unknown'))
    print(output, flush=True)


if __name__ == '__main__':
    a=argparse.ArgumentParser(__doc__); a.add_argument('--run-id', required=True); prepare(a.parse_args().run_id)
