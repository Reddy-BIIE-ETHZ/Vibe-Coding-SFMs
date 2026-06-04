from common_v4 import *

t0=start(); item='CL-12'
vals,_=parse_writeup_esfm_table()
r1=vals['ood040_e2s_mean']
ratio=r1/0.195
ok=ratio>100
r=finish(item,{'ood040_e2s_r1_pct':r1,'random_r1_pct':0.195,'ratio_x':ratio},'>100x','pass' if ok else 'fail',ok,None if ok else 'ratio not above 100x',t0)
print_result(r)
