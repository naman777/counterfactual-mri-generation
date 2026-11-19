"""Close local diagnostic rendering/reproducibility review; no model is promoted."""
import json

from scripts.v8_generator_dev.io import ROOT,digest,json_write,verify_inputs


def main():
    base=ROOT/'reports/v14_dropout_audit';out=base/'review_20261118_01';out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v14_dropout_audit.close. visual_review.json: '
        'manual review of all diagnostic figures; evidence.json: verified case/probe/exposure/algebra '
        'receipts; inputs.json: output/source identities; complete.json: local diagnostic scope. '
        'Human-readable entrypoint is ../report_20261118_01/RESULTS.md. No primary gate or model change.\n',encoding='utf-8')
    folders=['paired_diagnosis_20261118_01','case_verification_20261118_01','exposure_20261118_01',
             'epsilon_probe_20261118_01','probe_verification_20261118_01','shared_residual_algebra_20261118_01',
             'visuals_20261118_01','report_20261118_01']
    evidence={};inputs={}
    for name in folders:
        home=base/name
        receipt=json.loads((home/'complete.json').read_text())
        if not receipt['complete']:raise ValueError('Incomplete local diagnostic stage')
        verify_inputs(json.loads((home/'inputs.json').read_text()))
        evidence[name]=receipt
        for path in (home/'complete.json',home/'inputs.json'):
            inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    figures=['effects.png','patients.png','exposure.png','epsilon_response.png']
    visual={name:digest(base/'visuals_20261118_01'/name) for name in figures}
    json_write(out/'visual_review.json',dict(manually_viewed_all=True,figure_hashes=visual,
        observations='All judge labels/CIs and every source/seed heatmap row readable; source order retained; '
            'warnings 024 and 255 explicitly marked. Exposure bars show unique matched streams. '
            'Epsilon plots clearly labelled as surrogate fixed-input responses, not editing success.',
        clinical_anatomy_review=False))
    json_write(out/'evidence.json',evidence)
    for name in figures:
        path=base/'visuals_20261118_01'/name;inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    for path in (base/'report_20261118_01/RESULTS.md',base/'report_20261118_01/patient_contributions.csv',
        base/'shared_residual_algebra_20261118_01/NEXT_LOCAL_PROTOTYPE.md',ROOT/'scripts/v14_dropout_audit/close.py'):
        inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    json_write(out/'inputs.json',inputs)
    json_write(out/'complete.json',dict(complete=True,local_diagnosis_verified=True,all_figures_visually_reviewed=True,
        original_advancement_changed=False,new_reverse_sampling=False,new_training=False,paid_gpu_calls=0,
        full_sampler_implemented=False,clinical_preservation='unknown',default_promoted=False))
    with (base/'README.md').open('x',encoding='utf-8') as handle:
        handle.write('# Local dropout diagnosis\n\nStart with `report_20261118_01/RESULTS.md`; '
            'verification/rendering closure: `review_20261118_01/`.\n\n'
            '| Folder | Producer / contents |\n|---|---|\n'
            '| preparation_20261118_01 | `.prepare`: declared post-hoc plan, all controls/strata and bounded forward-only probe. |\n'
            '| checks_20261118_01 | unittest invocation: four pairing/warning/exposure checks. |\n'
            '| paired_diagnosis_20261118_01 | `.cases`: 69 contrasts, all 93 profiles, target/null/patient/replica/influence tables. |\n'
            '| case_verification_20261118_01 | `.verify_cases`: independent raw pivot and 360 direct-resampling primary intervals. |\n'
            '| exposure_20261118_01 | `.exposure`: every actual sample state and old/new stream comparison. |\n'
            '| epsilon_probe_20261118_01 | `.probe`: 2,160 fixed-source response observations with all raw epsilon arrays. |\n'
            '| probe_verification_20261118_01 | `.verify_probe`: independent raw metrics, source/condition/q checks, exact 30-noise replay. |\n'
            '| shared_residual_algebra_20261118_01 | `.algebra_audit`: two analytical tests and all-vector combiner checks; unvalidated sampler proposal. |\n'
            '| visuals_20261118_01 | `.visualise`: all-judge effects, all-source heatmaps, exposure and fixed-input response figures. |\n'
            '| report_20261118_01 | `.report`: readable diagnostic findings and all-patient signed contributions. |\n'
            '| review_20261118_01 | `.close`: final provenance/rendering review. |\n\n'
            'All module producers are under scripts.v14_dropout_audit. Work is local only; no training '
            'or reverse-sampling efficacy run, no paid app, no new test data or EchoTime edit. '
            'The original failed advancement requirements remain unchanged. The candidate combiner '
            'is a mathematical component, not a full sampler or demonstrated editing improvement.\n')
    print('LOCAL DIAGNOSIS AND FIGURE REVIEW COMPLETE',out,flush=True)


if __name__=='__main__':main()
