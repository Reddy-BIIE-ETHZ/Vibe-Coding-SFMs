from common_v3 import *

t0=start(); item='CL-7'
# expected log indices map
groups={'mmseqs_080':[5,6,7,8],'mmseqs_060':[10,11,12,13],'mmseqs_040':[15,16,17,18]}
vals_obs={}; source='training_logs'; ok_logs=True
for k,idxs in groups.items():
    arr=[]
    for i in idxs:
        txt=(ROOT/f'logs/esfm/esfm_full_62858369_{i}.out').read_text()
        m=re.findall(r'Pool-512 S→E R@1\s*[:=]\s*([0-9.]+)',txt)
        if m: arr.append(float(m[-1]))
    if len(arr)==4: vals_obs[k]=(float(np.mean(arr)),float(np.std(arr)),arr)
    else: ok_logs=False
if not ok_logs:
    vals,found=parse_writeup_esfm_table(); source='writeup_§3.2'
    if not found:
        r=finish(item,{'source':source},'58.6±2.2,53.9±2.5,40.6±3.0','skip',None,'log parse failed and writeup parse failed',t0); print_result(r); raise SystemExit
    vals_obs={'mmseqs_080':(vals['ood080_s2e_mean'],vals['ood080_s2e_sd'],[]),'mmseqs_060':(vals['ood060_s2e_mean'],vals['ood060_s2e_sd'],[]),'mmseqs_040':(vals['ood040_s2e_mean'],vals['ood040_s2e_sd'],[])}
exp={'mmseqs_080':(58.6,2.2),'mmseqs_060':(53.9,2.5),'mmseqs_040':(40.6,3.0)}
ok=all(abs(vals_obs[k][0]-exp[k][0])<=1.0 and abs(vals_obs[k][1]-exp[k][1])<=1.0 for k in exp)
r=finish(item,{'source_used':source,'observed':vals_obs},exp,'pass' if ok else 'fail',ok,None if ok else 'OOD means/sd outside tolerance',t0)
print_result(r)
