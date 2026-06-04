import subprocess, tempfile, pandas as pd
from common_v3 import *

t0=start(); item='SA-2'
try:
    import shutil
    if shutil.which('mmseqs') is None:
        raise RuntimeError('mmseqs not installed in environment')
    # proxy: if clustering succeeds, use criterion cluster file has one cluster per representative; true inter-cluster identity check unavailable without align-all
    r=finish(item,'not implemented exact inter-cluster identity computation in sandbox','max inter-cluster identity < 0.8','skip',None,'mmseqs proxy unsupported without pairwise alignment pass',t0)
except Exception as e:
    r=finish(item,str(e),'max inter-cluster identity < 0.8','skip',None,'could not execute mmseqs check',t0)
print_result(r)
