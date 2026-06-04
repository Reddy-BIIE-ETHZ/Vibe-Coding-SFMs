from common_v3 import *
from pathlib import Path

t0=start(); item='SA-6'
code=read_text('src/calm/encoder/eval_esfm_pool512.py')
logic='epoch_of(best_ckpt) == 0' in code
vals={}
for i in range(20):
    txt=(ROOT/f'logs/esfm/esfm_full_62858369_{i}.out').read_text()
    vals[i]=parse_best_val_epoch(txt)
expected={4:0,9:0,14:0,19:0}
ok_logic=logic
ok_vals=all(vals[k]==v for k,v in expected.items())
ok=ok_logic and ok_vals
r=finish(item,{'exclusion_logic_present':logic,'best_val_epoch_by_index':vals},'logic present and indices 4,9,14,19 epoch 0','pass' if ok else 'fail',ok,None if ok else 'mismatch in exclusion logic or epochs',t0)
print_result(r)
