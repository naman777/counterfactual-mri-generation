"""Prospectively fixed scientific settings; no paid approval is encoded here."""
import copy
import hashlib
import json

MODES=('coarse_control','coarse_fine','learned_zero')
TRAINING_SEEDS=(583301,583302,583303)
GUIDE_CONTROLS=('correct','zero','absent','wrong','coarse_only','fine_only')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def make_protocol(parent_protocol):
    p=parent_protocol
    return dict(format='replicated_detail_v1',view=p['view'],
        parent=p['parents']['additive'],parent_sha256=p['parent_hashes']['additive'],parent_epoch=130,
        backbone='additive',training_seeds=list(TRAINING_SEEDS),modes=list(MODES),
        training=dict(epochs=20,block_epochs=5,batch_size=32,learning_rate=1e-4,weight_decay=.01,
            gradient_clip=1.,condition_drop_probability=.1,structure_proxy_weight=.1,
            structure_proxy_max_timestep=300,adapter_width=16,max_same_batch_precision_retries=8,
            guide='6x64 coarse/coarse versus coarse/fine versus learned zero',
            loss='Unchanged full-timestep noise MSE plus coarse source-guide consistency for t<=300',
            final_weights='Raw common final epoch, no validation checkpoint selection',
            sampling='Historical patient/cell balancing, independent seed/data/noise/drop streams',
            precision='FP16 autocast plus bounded identical-draw scaler backoff; persistent failure fatal'),
        evaluation=copy.deepcopy(p['parent_config']['evaluation']),
        primary_axis='manufacturer',secondary_axis='field_strength',
        controls=dict(manufacturer={m:list(GUIDE_CONTROLS) if m!='learned_zero' else ['correct','absent'] for m in MODES},
                      field_strength={m:['correct'] for m in MODES}),
        mask_policy=dict(training_masking=False,evaluation_masking=False,
            region_mask='noise_aware_v1, fixed constants and train-only warning cutoffs',
            warning_cases='Retain every source, flag and report separately; no favourable-case exclusion',
            invalid_region='Keep whole-image/judge results, set regional metrics missing; advancement unavailable if a primary region is invalid',
            masking_ablation='Separate later protocol; not bundled into this guide comparison'),
        analysis=dict(bootstrap_resamples=10000,bootstrap_seed=583341,
            uncertainty='Paired crossed bootstrap of patients and training seeds; generation seeds averaged within each crossed cell; per-training-seed patient intervals also required',
            primary_contrast='coarse_fine minus capacity-matched coarse_control, correct guide, manufacturer, all requests retained',
            gates=[dict(metric='interior_mad',control='target',upper_less_than=0.),
                   dict(metric='interior_hf_residual_mad',control='target',upper_less_than=0.),
                   dict(metric='interior_mad',control='null',upper_less_than=.005),
                   *[dict(metric='paired_hit_delta',control='target',judge=f'heldout_orientation_robust_seed{j}',
                          lower_greater_than=-.05) for j in range(3)]],
            family_alpha=.05,family_size=6,interval_quantiles=[.05/(2*6),1-.05/(2*6)],
            margins='Predeclared research tolerances: .005 normalised null-MAD and five percentage points editing; not clinical equivalence/safety margins',
            training_replica_rule='All three replica point differences must favour lower target interior MAD and HF residual; all named-judge editing differences must be >=-.05 in each replica',
            safeguards='Retain original direction/support and other-axis stability constraints on selection seed, evaluated per training replica; field-strength remains exploratory/infeasible if support missing',
            decision='Advancement means a development research candidate for later independent confirmation, never clinical approval or default promotion',
            primary_missingness='Any missing crossed cell, replica, generation seed, judge, source or primary regional metric makes the advancement decision unavailable'),
        scope=dict(reused_validation=True,test_pixels_used=False,echo_time_edited=False,
            acquisition_invariance=False,clinical_preservation='unknown',
            historical_defaults_changed=False,paid_approval=False,
            mask_warning_cutoffs='reports/v9_detail/mask_audit_20261117_01/thresholds.json'))


def validate(p):
    if p['format']!='replicated_detail_v1' or p['backbone']!='additive' or p['parent_epoch']!=130:
        raise ValueError('Frozen additive final-parent protocol required')
    if p['training_seeds']!=list(TRAINING_SEEDS) or p['modes']!=list(MODES):
        raise ValueError('All three prospective replicas/modes required')
    if p['training']['batch_size']!=32 or p['training']['epochs']!=20:
        raise ValueError('Matched batch/exposure changed')
    if p['scope']['test_pixels_used'] or p['scope']['echo_time_edited'] or p['mask_policy']['evaluation_masking']:
        raise ValueError('No test, EchoTime edits or conflated mask experiment')
    if len(p['analysis']['gates'])!=6 or p['analysis']['family_size']!=6:
        raise ValueError('Predeclared six-gate family required')
    if p['scope']['paid_approval']:
        raise ValueError('Scientific protocol cannot grant paid permission')


def copy_key(seed,mode):
    if seed not in TRAINING_SEEDS or mode not in MODES:raise ValueError('Unknown copy')
    return f'seed{seed}__{mode}'
