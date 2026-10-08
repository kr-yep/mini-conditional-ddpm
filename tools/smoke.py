"""Check fake-data training, checkpoint reload, and sampling without downloading data."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
output_root = root / 'outputs'
output_root.mkdir(exist_ok=True)
out = Path(tempfile.mkdtemp(prefix='smoke-', dir=output_root))
assert out.resolve().is_relative_to(output_root.resolve())
config = root / 'final_project/configs/smoke.yaml'
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
commands = [
    [sys.executable, 'final_project/src/train_ddpm.py', '--config', str(config), '--fake-data', '--output-dir', str(out)],
    [sys.executable, 'final_project/src/sample_ddpm.py', '--config', str(config), '--checkpoint', str(out / 'checkpoints/latest.pt'), '--output-dir', str(out / 'samples')],
]
for i, command in enumerate(commands):
    with (out / f'step-{i}.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
images = list((out / 'samples/images').rglob('*.png'))
if len(images) != 10:
    raise RuntimeError(f'Expected ten class samples, found {len(images)}; inspect {out}')
print(json.dumps({'status': 'passed', 'training': '8 fake images, 1 epoch, CPU', 'sample_images': len(images),
                  'output': str(out.relative_to(root)), 'scope': 'Pipeline mechanics only; no image-quality claim.'}))
