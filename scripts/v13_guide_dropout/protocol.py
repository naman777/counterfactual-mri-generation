"""Bind the proposed single-factor change and all ten prospective requirements."""
import copy

from scripts.v10_detail.protocol import validate as validate_reference, canonical_hash
from scripts.v11_guide_audit.guide_dropout import STATE_NAMES, STATE_PROBABILITIES

MODES = ('coarse_control','coarse_fine')
FORMAT = 'guide_dropout_full_protocol_v1'
CHECKPOINT_FORMAT = 'guide_dropout_full_checkpoint_v1'


def policy():
    return dict(states=list(STATE_NAMES),probabilities=list(STATE_PROBABILITIES),
                rng='torch.multinomial, independent device generator per applied-step cursor',
                seed_offset=5000000,step_cursor='zero-based applied updates before forward',
                wrong_patient_training=False,zero_input_is_absent=False)


def make_protocol(reference):
    validate_reference(reference)
    gates = copy.deepcopy(reference['analysis']['gates'])
    for gate in gates:
        gate['comparison'] = 'new_fine_minus_new_coarse'
    gates += [dict(metric='paired_hit_delta',control='target',judge=f'heldout_orientation_robust_seed{j}',
                   comparison='new_fine_minus_old_fine',lower_greater_than=-.05) for j in range(3)]
    gates += [dict(metric='interior_mad',control='target',comparison='new_fine_minus_old_fine',upper_less_than=.005)]
    return dict(format=FORMAT,reference_protocol=copy.deepcopy(reference),
                reference_protocol_sha256=canonical_hash(reference),modes=list(MODES),
                guide_dropout=policy(),analysis=dict(gates=gates,bootstrap_resamples=10000,
                bootstrap_seed=583341,family_size=10,family_alpha=.05,interval_quantiles=[.0025,.9975],
                no_default_promotion=True,all_original_safeguards_retained=True),
                expected=dict(copies=6,epochs=20,updates_per_copy=2780,updates=16680,
                              evaluation_profiles=42,cases=1890,named_grades=17010),
                old_baseline_receipt='reports/v10_detail/guide_replicates_20261117_01_training/complete.json',
                paid_authority_in_scientific_protocol=False)


def validate(p):
    if not isinstance(p,dict) or p.get('format') != FORMAT:
        raise ValueError('Distinct full-exposure dropout protocol required')
    reference = p['reference_protocol']
    validate_reference(reference)
    if p != make_protocol(reference):
        raise ValueError('Dropout policy, gate family or declared design changed')
    if reference['training']['block_epochs'] != 5 or reference['training']['condition_drop_probability'] != .1:
        raise ValueError('Historical matched exposure and acquisition dropout required')


def copy_key(seed,mode):
    if mode not in MODES or seed not in (583301,583302,583303):
        raise ValueError('Unknown prospective copy')
    return f'seed{seed}__{mode}'


def evaluation_protocol(p):
    validate(p)
    result=copy.deepcopy(p['reference_protocol'])
    result['modes']=list(MODES)
    result['controls']={axis:{mode:routes for mode,routes in arms.items() if mode in MODES}
                        for axis,arms in result['controls'].items()}
    return result
