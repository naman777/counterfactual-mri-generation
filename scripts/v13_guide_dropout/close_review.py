"""Record observed terminal billing and the completed manual figure review."""
import json
import subprocess
import sys
from decimal import Decimal

from scripts.v8_generator_dev.io import ROOT,digest,json_write,verify_inputs


def main():
    base=ROOT/'reports/v13_guide_dropout'
    report=base/'report_20261118_01';analysis=base/'analysis_20261118_01'
    verify_inputs(json.loads((report/'inputs.json').read_text()))
    verify_inputs(json.loads((analysis/'inputs.json').read_text()))
    training=json.loads((base/'training_verified_20261118_01/complete.json').read_text())
    checked=json.loads((base/'verified_20261118_01/complete.json').read_text())
    outcome=json.loads((analysis/'advancement.json').read_text())
    final=json.loads((base/'results_20261118_01/root/complete.json').read_text())
    if not training['complete'] or not checked['saved_outputs_verified'] or not final['workflow_success']:
        raise ValueError('Complete verified experiment required')
    out=base/'final_review_20261118_01';out.mkdir(parents=True,exist_ok=False)
    (out/'README.md').write_text('Producer: scripts.v13_guide_dropout.close_review. RESULTS.md: '
        'final interpretation and evidence entrypoint; billing_after.json/apps_after.json: current '
        'read-only provider observations; budget.json: observed delta and conservative worker estimate, '
        'not an exact per-app invoice; visual_review.json: manual rendering review of all four figures; '
        'inputs.json: bound evidence; complete.json: execution/verification/report review distinct '
        'from scientific advancement. No new cloud job.\n',encoding='utf-8')
    observed={}
    for key,command in [('billing_after',['billing','summary','--for','2026-11']),('apps_after',['app','list'])]:
        result=subprocess.run([sys.executable,'-m','modal',*command,'--json','--profile','namankundra7'],
            cwd=ROOT,capture_output=True,text=True,check=True,timeout=60)
        observed[key]=json.loads(result.stdout);json_write(out/(key+'.json'),observed[key])
    if any(app['state']!='stopped' for app in observed['apps_after']):raise ValueError('An app is still active')
    before=json.loads((base/'launch_20261118_01/billing_before.json').read_text())
    after=observed['billing_after'];plan=json.loads((base/'preparation_20261118_01/plan.json').read_text())
    delta=Decimal(after['metered_cost'])-Decimal(before['metered_cost'])
    credits=abs(Decimal(after['adjustments']['credits']))-abs(Decimal(before['adjustments']['credits']))
    estimate=final['seconds']*plan['price_with_margin']+plan['reserve_usd']
    budget=dict(observed_metered_delta_usd=float(delta),observed_credit_debit_delta_usd=float(credits),
        metering_may_settle=True,exact_per_app_invoice=False,worker_seconds=final['seconds'],
        conservative_worker_estimate_with_reserve_usd=estimate,working_limit_usd=12,
        user_total_authority_usd=13,no_active_apps=True,allowance_closed=True,
        unused_remainder_automatically_reassigned=False,no_new_cloud_job_started=True)
    json_write(out/'budget.json',budget)
    figures=['gates.png','patients_01.png','patients_02.png','patients_03.png']
    visual=dict(manually_viewed_all=True,figures={name:digest(report/name) for name in figures},
        rendering='Titles, all ten gate intervals and all fifteen source-patient rows are readable; none omitted.',
        observations='Substantial source-to-edit texture and boundary changes remain visible, including warning sources. '
                     'Visual rendering inspection is not expert anatomy/lesion validation.',
        clinical_anatomy_review=False,scientific_numbers_changed=False)
    json_write(out/'visual_review.json',visual)
    passed=sum(g['available'] and g['interval_passed'] and g['replica_direction_passed'] for g in outcome['gates'])
    failed=[dict(comparison=g['rule']['comparison'],judge=g['judge'],metric=g['rule']['metric'])
            for g in outcome['gates'] if not (g['available'] and g['interval_passed'] and g['replica_direction_passed'])]
    lines=['# Completed guide-state dropout experiment','',
        'Execution and independent verification completed successfully. The candidate **does not meet '
        'the frozen advancement rule** and is not promoted.','',
        f"- Six adapters finish 20 epochs each; all {training['actual_applied_updates']:,} applied updates are audited.",
        f"- All 42 profiles / {checked['cases']:,} saved cases / {checked['grades']:,} named-judge/control grades pass independent verification.",
        f"- {passed} of 10 simultaneous requirements pass; all three original support/correspondence safeguards pass.",
        '- Three interval safeguards fail: robust seed0/seed1 editing versus the new coarse guide, '
        'and robust seed0 editing versus the cached old fine guide. All per-replica point rules pass; '
        'interval uncertainty prevents advancement. Failure to establish noninferiority is not proof of harm.','',
        'The new fine guide is closer to source than its matched new coarse guide: strict-interior MAD '
        'difference -0.009871 [simultaneous CI -0.011734, -0.007891]. Against cached old fine guidance, '
        'source-MAD difference is +0.000195 [-0.001244, +0.001653]. These proxies do not establish anatomy preservation.','',
        'Named editing changes against cached old fine guidance (percentage points, simultaneous CIs): '
        'heldout_orientation_robust_seed0 +0.74 [-8.89, +14.81]; seed1 0.00 [0.00, 0.00]; '
        'seed2 +2.96 [0.00, +11.11]. None has a strictly positive lower interval. '
        'Do not claim a demonstrated editing improvement from the favourable point estimates.','',
        'The fixed development cohort is reused. Field strength stays exploratory; source warnings '
        'and all adverse controls are retained. EchoTime stays source. Clinical anatomy/lesion preservation, '
        'causal scanner realism and untouched confirmation remain unknown. All four report figures were '
        'manually inspected for rendering; that review is not clinical validation.','',
        f"The cloud worker used {final['seconds']/3600:.3f} hours. The provider billing summary currently shows "
        f"US${float(credits):.2f} additional applied credits; this is a workspace observation and may settle, "
        f"not an exact per-app invoice. The conservative worker estimate plus internal reserve is US${estimate:.2f}, "
        'below the US$12 working limit and US$13 authority. No Modal app is active; the allowance for this '
        'experiment is closed. No new paid run is launched.','',
        'Next work should start locally: inspect the failed named-judge cells and actual guide-state '
        'exposure before designing another change. No extra epochs, default change or new cloud job is '
        'implied by this report.','',
        'Detailed unchanged tables and figures: `../report_20261118_01/RESULTS.md`; '
        'full strata/replica intervals: `../analysis_20261118_01/`.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=[base/'training_verified_20261118_01/complete.json',base/'verified_20261118_01/complete.json',
        analysis/'advancement.json',report/'RESULTS.md',base/'results_20261118_01/root/complete.json',
        base/'launch_20261118_01/billing_before.json',out/'billing_after.json',out/'apps_after.json',
        out/'budget.json',out/'visual_review.json',ROOT/'scripts/v13_guide_dropout/close_review.py']
    paths += [report/name for name in figures]
    json_write(out/'inputs.json',{path.relative_to(ROOT).as_posix():digest(path) for path in paths})
    json_write(out/'complete.json',dict(complete=True,execution_verified=True,visual_review_completed=True,
        research_advancement=False,gates_passed=passed,gates=10,failed=failed,default_promoted=False,
        budget_observation_recorded=True,no_active_modal_apps=True,allowance_closed=True,
        clinical_preservation='unknown',fresh_test=False))
    print(json.dumps(dict(final_review=out.relative_to(ROOT).as_posix(),gates_passed=passed,
        research_advancement=False,budget=budget),indent=2),flush=True)


if __name__=='__main__':main()
