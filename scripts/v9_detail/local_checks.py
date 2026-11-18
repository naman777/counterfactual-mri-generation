"""Run and archive the local preparation regression checks."""
import argparse
import unittest
from contextlib import redirect_stdout,redirect_stderr

from scripts.v9_detail.common import start,finish
from scripts.v8_generator_dev.io import json_write


def run(run_id):
    output,inputs=start(run_id,'scripts.v9_detail.local_checks','Local preparation regression checks')
    json_write(output/'protocol.json',dict(test_modules=['scripts.v9_detail.test_local'],
        scope='Synthetic mask, source-decoder, fine-control and patient-uncertainty engineering checks'))
    suite=unittest.defaultTestLoader.loadTestsFromName('scripts.v9_detail.test_local')
    with (output/'tests.txt').open('x',encoding='utf-8') as handle,redirect_stdout(handle),redirect_stderr(handle):
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError('Local checks failed; report retained')
    finish(output,inputs,tests=result.testsRun,failures=0)
    print('LOCAL CHECKS PASS',result.testsRun,output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-id',required=True)
    run(p.parse_args().run_id)
