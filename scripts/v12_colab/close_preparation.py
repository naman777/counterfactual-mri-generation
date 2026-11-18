"""Record the completed local compatibility preparation without claiming a Colab run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'reports/v12_colab'
PACKAGE = 'compatibility_package_20261117_01'
CHECK = 'local_validation_20261118_01'


def main():
    validation = json.loads((BASE/CHECK/'complete.json').read_text())
    synthetic = json.loads((BASE/CHECK/'synthetic/result.json').read_text())
    if not validation['complete'] or not synthetic['complete']:
        raise ValueError('Cannot close incomplete local checks')
    with (BASE/'README.md').open('x',encoding='utf-8') as f:
        f.write('# Colab compatibility preparation\n\n'
            'Producer: scripts.v12_colab.close_preparation (index only).\n\n'
            f'- `{PACKAGE}/`: source-only ZIP and Colab notebook, exact hashes and isolated import receipt; '
            'producer `python -m scripts.v12_colab.build_compatibility`.\n'
            f'- `{CHECK}/`: notebook syntax, isolated imports, modified-source rejection and actual disposable '
            'local batch-five updates; producer `python -m scripts.v12_colab.validate_package`.\n\n'
            'The user screenshot shows Python 3.13.16, PyTorch 2.14.1+cu130 and Tesla T4. '
            'It confirms GPU access only. Colab execution of this package and batch-32 capacity remain unverified. '
            'No MRI pixels/trained weights are packaged, no software is installed, no Drive permission is requested, '
            'and no full training is launched. The notebook downloads its report before the ephemeral runtime expires. '
            'This is not the pending full-exposure dropout trainer or a validated resume workflow.\n')
    marker = 'Colab source-only compatibility preparation (2026-11-18)'
    common = (f'Outputs: `counterfactual-mri-generation/reports/v12_colab/{PACKAGE}/` and '
              f'`{CHECK}/`. Notebook and source ZIP are built; isolated imports, modified-source rejection, '
              'notebook cell syntax and four disposable local batch-five updates pass. Both guide modes '
              'exercise all five states with matched synthetic noise/gates, changed adapters and frozen '
              'random parent/conditioner. No data or trained checkpoint is uploaded or included. '
              'Colab T4 batch-32 check, exact trained-parent/data parity, full-exposure trainer and strict '
              'interrupted resume remain pending. No efficacy/default/publication change or new cloud run.\n')
    docs = ROOT.parent/'improvements'
    entries = {
        'GENERATOR_DEV_PROGRESS.md':'\n## '+marker+'\n\n'+common,
        'DECISIONS.md':'\n## '+marker+'\n\n'+common+
            'Use the installed Colab stack for a bounded synthetic compatibility check first; do not '
            'silently change dependencies or treat it as equivalent to historical experiments. '
            'No previous paid allowance is reopened. Full-run scope/time/compute units require a separate review.\n',
        'ROADMAP.md':'\n## '+marker+'\n\n- [x] Prepare and locally verify the source-only Colab notebook/package. '+common+
            '- [ ] Receive and review the actual Colab T4 batch-32 compatibility report.\n',
    }
    for name,text in entries.items():
        path = docs/name
        if marker in path.read_text(encoding='utf-8'):
            raise ValueError('Progress already recorded')
        with path.open('a',encoding='utf-8') as f:
            f.write(text)
    status = docs/'STATUS.md'
    text = status.read_text(encoding='utf-8')
    first,rest = text.split('\n',1)
    status.write_text(first+'\n\n## '+marker+'\n\n'+common+'\n'+rest,encoding='utf-8')
    print('Preparation recorded; Colab runtime/full training still pending.')


if __name__ == '__main__':
    main()
