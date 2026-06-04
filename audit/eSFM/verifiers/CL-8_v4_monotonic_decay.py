from common_v4 import *

t0=start(); item='CL-8'
vals,_=parse_writeup_esfm_table()
s2e=[vals['id_s2e_mean'],vals['ood080_s2e_mean'],vals['ood060_s2e_mean'],vals['ood040_s2e_mean']]
e2s=[vals['id_e2s_mean'],vals['ood080_e2s_mean'],vals['ood060_e2s_mean'],vals['ood040_e2s_mean']]
ok=all(s2e[i]>s2e[i+1] for i in range(3)) and all(e2s[i]>e2s[i+1] for i in range(3))
r=finish(item,{'s2e':s2e,'e2s':e2s},'strict monotonic decay in both directions','pass' if ok else 'fail',ok,None if ok else 'non-monotonic values',t0)
print_result(r)
