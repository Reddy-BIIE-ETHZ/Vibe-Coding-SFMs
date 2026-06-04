from common_v4 import *
import yaml

t0=start(); item='CL-4'
cfg=yaml.safe_load((ROOT/'configs/train/encoder/esfm_full.yaml').read_text())
expected={
'learning_rate':0.001,'weight_decay':0.2,'scheduler':'CosineAnnealingWarmRestarts','warmup_ratio':0.1,
'optimizer':'AdamW','optimizer_T0':20,'optimizer_Tmult':1,'num_epochs':100,'batch_size':64,
'batch_size_val':64,'checkpoint_phase':'val','checkpoint_metric':'pred_acc','checkpoint_mode':'max'
}
obs={k:cfg.get(k) for k in expected}
ok=all(obs[k]==v for k,v in expected.items())
r=finish(item,obs,expected,'pass' if ok else 'fail',ok,None if ok else 'one or more config parameters differ',t0)
print_result(r)
