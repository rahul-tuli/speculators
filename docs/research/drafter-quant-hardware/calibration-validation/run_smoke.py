"""Small plumbing checks only, never experiment measurements."""
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = Path('/data/fast/drafter-quant/data/perfectblend_2048_prepared')
OUTPUT = Path('/data/fast/drafter-quant/validation/calibration-input-20260921')
OUTPUT.mkdir(parents=True, exist_ok=True)
(HERE / 'packages.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
(HERE / 'speculators.patch').write_text(subprocess.check_output(['git','diff','HEAD'], text=True))
(HERE / 'git_head.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'], text=True))
for name in ('vllm_command.txt', 'vllm.patch', 'checkpoint_sha256.txt'):
    shutil.copy2(CACHE / name, HERE / ('capture_' + name))
source = HERE / 'source'
source.mkdir(exist_ok=True)
for name in ('quantize.py','calibration.py'):
    shutil.copy2(ROOT / 'local/drafter-quant/dquant' / name, source / name)

def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

model_root = Path('/data/fast/hf_cache/hub/models--RedHatAI--Qwen3-8B-speculator.dflash/snapshots/1a11b170eb65c8a62c80ecd01dfe22a5907298e6')
(HERE / 'drafter_checkpoint_sha256.txt').write_text(''.join(f'{digest_file(p)}  {p}\n' for p in sorted(model_root.glob('*')) if p.is_file() and p.suffix in ('.json','.safetensors')))
results=[]
for scheme in ('fp8_w8a8','nvfp4_w4a4'):
    for calibration in ('real','random'):
        name=f'{scheme}-{calibration}'
        out=OUTPUT/name
        if out.exists():
            raise RuntimeError(f'refusing to overwrite {out}')
        cmd=[sys.executable,'-m','dquant.quantize','--scheme',scheme,
             '--model','RedHatAI/Qwen3-8B-speculator.dflash','--processor','Qwen/Qwen3-8B',
             '--calib-data',str(CACHE) if calibration=='real' else 'random',
             '--num-calibration-samples','1','--seq-len','512','--seed','0',
             '--output-dir',str(out)]
        if scheme=='nvfp4_w4a4':
            cmd += ['--nvfp4-weight-observer','nvfp4_expanded_mse']
        env={**os.environ, 'PYTHONPATH':str(ROOT/'local/drafter-quant'),
             'HF_HOME':'/data/fast/hf_cache','HF_HUB_CACHE':'/data/fast/hf_cache/hub',
             'HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'}
        (HERE/f'{name}_command.txt').write_text(
            'HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 '
            'HF_HOME=/data/fast/hf_cache HF_HUB_CACHE=/data/fast/hf_cache/hub '
            f'PYTHONPATH={ROOT}/local/drafter-quant '+shlex.join(cmd)+'\n')
        start=time.monotonic()
        with (HERE/f'{name}.log').open('w') as log:
            run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        result={'name':name,'exit_code':run.returncode,'elapsed_seconds':time.monotonic()-start,
                'output_dir':str(out)}
        print(result,flush=True)
        results.append(result)
        (HERE/'smoke_results.json').write_text(json.dumps(results,indent=2)+'\n')
        if run.returncode:
            sys.exit(run.returncode)
        (HERE/f'{name}_checkpoint_sha256.txt').write_text(''.join(
            f'{digest_file(p)}  {p}\n' for p in sorted(out.glob('*'))
            if p.is_file() and p.suffix in ('.json','.safetensors')))
