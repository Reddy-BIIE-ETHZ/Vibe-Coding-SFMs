from common_v3 import *

t0=start(); item='CL-11'
vals=[]
for i in [0,1,2,3]:
    vals.append(parse_best_val_epoch((ROOT/f'logs/esfm/esfm_full_62858369_{i}.out').read_text()))
ok=all(v is not None and 11<=v<=45 for v in vals)
r=finish(item,{'best_val_epochs':vals},'all in [11..45], expected near {15,30,15,40}','pass' if ok else 'fail',ok,None if ok else 'epoch outside tolerance or missing',t0)
print_result(r)
