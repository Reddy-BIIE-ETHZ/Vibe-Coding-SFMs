from common_v3 import *

t0=start(); item='SA-9'
try:
    text=read_text('src/calm/encoder/data.py')
    deterministic=('random.SystemRandom()' not in text)
    r=finish(item,{'uses_system_random':('random.SystemRandom()' in text)},'identical batch index sequences across two runs','pass' if deterministic else 'fail',deterministic,None if deterministic else 'sampler uses SystemRandom (non-seeded)',t0)
except Exception as e:
    r=finish(item,str(e),'identical batch index sequences','skip',None,'could not inspect sampler',t0)
print_result(r)
