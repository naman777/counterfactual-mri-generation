"""Require actual recorded optimisation exposure before the unchanged ten-gate analysis."""
import argparse
import json

from scripts.v8_generator_dev.io import ROOT,digest,json_write,project_path,verify_inputs
from scripts.v13_guide_dropout.analyse import analyse as original_analyse


def analyse(prepared,results,verified,training_verified,run_id):
    training_verified=project_path(training_verified)
    evidence=json.loads((training_verified/'complete.json').read_text())
    if not evidence['complete'] or evidence['actual_applied_updates']!=16680 or evidence['copies']!=6:
        raise ValueError('Independently audited actual training exposure required')
    ledger=json.loads((training_verified/'inputs.json').read_text());verify_inputs(ledger)
    training=json.loads((project_path(results)/'complete.json').read_text())['training']
    if training+'/complete.json' not in ledger:raise ValueError('Different training receipt was audited')
    out=original_analyse(prepared,results,verified,run_id)
    json_write(out/'training_provenance.json',dict(verified=training_verified.relative_to(ROOT).as_posix(),
        complete_sha256=digest(training_verified/'complete.json'),inputs_sha256=digest(training_verified/'inputs.json'),
        wrapper_sha256=digest(__file__),actual_applied_updates=16680))
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ('prepared','results','verified','training-verified','run-id'):ap.add_argument('--'+name,required=True)
    analyse(**vars(ap.parse_args()))
