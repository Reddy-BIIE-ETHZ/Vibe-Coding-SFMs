from common_v4 import *
import numpy as np

t0=start(); item='SA-5'
rng=np.random.default_rng(42)
trials=100; pool=512;dim=512
hits={1:[],5:[],10:[]}
for _ in range(trials):
    q=rng.normal(size=(pool,dim)); q=q/np.linalg.norm(q,axis=1,keepdims=True)
    t=rng.normal(size=(pool,dim)); t=t/np.linalg.norm(t,axis=1,keepdims=True)
    sims=q@t.T
    for i in range(pool):
        ranks=np.argsort(-sims[i])
        pos=int(np.where(ranks==i)[0][0])+1
        for k in hits:
            hits[k].append(pos<=k)
obs={f'R@{k}':100*np.mean(v) for k,v in hits.items()}
exp={'R@1':100/pool,'R@5':500/pool,'R@10':1000/pool}
ok=all(abs(obs[k]-exp[k])<0.1 for k in obs)
r=finish(item,obs,exp,'pass' if ok else 'fail',ok,None if ok else 'empirical deviates >0.1%',t0)
print_result(r)
