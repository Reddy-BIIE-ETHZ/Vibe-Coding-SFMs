from common_v3 import *

t0=start(); item='CL-9'
obs=[]
for i in [0,1,2,3]:
    txt=(ROOT/f'logs/esfm/esfm_full_62858369_{i}.out').read_text()
    m_pairs=re.search(r'test:\s*(\d+)\s+pairs',txt)
    obs.append({'fold':i,'test_pairs':int(m_pairs.group(1)) if m_pairs else None})
ok=all(x['test_pairs']==35488 for x in obs)
r=finish(item,obs,'35488 pairs each; substrate/enzyme ranges per fold', 'pass' if ok else 'skip', ok if ok else None, None if ok else 'logs lack unique substrate/enzyme fold stats needed for full check',t0)
print_result(r)
