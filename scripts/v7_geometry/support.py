"""Training-only condition support, never a causal or clinical approval gate."""
import numpy as np


def request_support(train,manufacturer,field_strength,echo_time_norm,min_patients=5,tolerance=1e-5):
    if manufacturer not in [0,1] or field_strength not in [0,1] or not np.isfinite(echo_time_norm):
        raise ValueError('Invalid target condition')
    cell=train[(train.manufacturer_label==manufacturer)&(train.field_strength_label==field_strength)]
    echo=cell.echo_time_norm.to_numpy(float);patients=int(cell.patient_id.nunique())
    within=bool(len(echo) and echo.min()-tolerance<=echo_time_norm<=echo.max()+tolerance)
    exact=bool(len(echo) and np.min(np.abs(echo-echo_time_norm))<=tolerance)
    if not patients:status='missing_target_cell'
    elif patients<min_patients:status='sparse_target_cell'
    elif not within:status='outside_target_echo_range'
    elif not exact:status='interpolation_only'
    else:status='observed_condition'
    return dict(condition_support_status=status,target_train_patients=patients,
                echo_within_target_range=within,echo_exact_seen=exact,
                nearest_echo_norm_distance=float(np.min(np.abs(echo-echo_time_norm))) if len(echo) else None,
                causal_edit_identified=False,clinical_approval=False,
                engineering_threshold_patients=min_patients)


def require_research_support(result,allow_unsupported=False):
    if result['condition_support_status']!='observed_condition' and not allow_unsupported:
        raise ValueError('Unsupported/unverified request: '+result['condition_support_status']+
                         '; an explicit research override is required')
    return 'research_only_not_clinically_validated'
