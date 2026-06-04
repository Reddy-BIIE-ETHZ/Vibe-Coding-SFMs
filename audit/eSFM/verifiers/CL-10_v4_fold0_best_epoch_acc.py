from common_v4 import *

t0=start(); item='CL-10'
text=(ROOT/'logs/esfm/esfm_full_62858369_0.out').read_text()
best=parse_best_val_epoch(text)
tr,va,te=parse_epoch_metrics(text)
train=tr.get(best); val=va.get(best); test=te.get(best)
ok=(best is not None and 11<=best<=15 and train is not None and abs(train*100-97.8)<=1.0 and abs(val*100-96.1)<=1.0 and abs(test*100-96.1)<=1.0)
r=finish(item,{'best_epoch':best,'train_acc_pct':None if train is None else train*100,'val_acc_pct':None if val is None else val*100,'test_acc_pct':None if test is None else test*100},{'epoch':'11..15','train':97.8,'val':96.1,'test':96.1},'pass' if ok else 'fail',ok,None if ok else 'metrics outside tolerance',t0)
print_result(r)
