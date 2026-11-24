#!/usr/bin/env python3
"""Build the pinned renderer into this project's output directory, without upgrading the training environment."""
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
REVISION='253ac4fcea7de5f396371124af597e6cc957bfae'
SOURCE=ROOT/'output/dependencies/nvdiffrast'
RUNTIME=ROOT/'output/dependencies/nvdiffrast_runtime'
if RUNTIME.is_dir():
    print('Renderer already installed:',RUNTIME)
    raise SystemExit(0)
if not SOURCE.exists():
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','clone','https://github.com/NVlabs/nvdiffrast.git',str(SOURCE)],check=True)
    subprocess.run(['git','-C',str(SOURCE),'checkout',REVISION],check=True)
actual=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
if actual!=REVISION:raise ValueError('Renderer source revision differs from pinned experiment version')
toolkit=Path(os.environ.get('FULL_MASK_CUDA_TOOLKIT','/home/p3/miniforge3/envs/lerobot-xense'))
if not (toolkit/'bin/nvcc').is_file():raise FileNotFoundError('Set FULL_MASK_CUDA_TOOLKIT to an existing CUDA 12.8 toolkit')
prefix=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia'
includes=[str(toolkit/'targets/x86_64-linux/include')]+[str(p) for p in prefix.glob('*/include')]
libs=[str(toolkit/'targets/x86_64-linux/lib')]+[str(p) for p in prefix.glob('*/lib')]
env=dict(os.environ);env.update(CUDA_HOME=str(toolkit),PATH=str(toolkit/'bin')+':'+env['PATH'],CXX='/usr/bin/g++',CC='/usr/bin/gcc',MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='12.0',OPENBLAS_NUM_THREADS='1',CPATH=':'.join(includes),LIBRARY_PATH=':'.join(libs),NVCC_PREPEND_FLAGS=' '.join('-I'+p for p in includes))
subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','--target',str(RUNTIME),str(SOURCE)],env=env,check=True)
print('Renderer installed:',RUNTIME)
