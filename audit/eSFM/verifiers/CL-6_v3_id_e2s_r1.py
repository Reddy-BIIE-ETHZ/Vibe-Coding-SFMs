from common_v3 import *

t0=start(); item='CL-6'
log_vals=[]
for i in [0,1,2,3]:
    txt=(ROOT/f'logs/esfm/esfm_full_62858369_{i}.out').read_text()
    m=re.findall(r'Pool-512 E→S R@1\s*[:=]\s*([0-9.]+)',txt)
    if m: log_vals.append(float(m[-1]))
source='training_logs'
if len(log_vals)==4:
    mean=float(np.mean(log_vals)); sd=float(np.std(log_vals))
else:
    vals,found=parse_writeup_esfm_table(); source='writeup_§3.2'
    if not found:
        r=finish(item,{'source':source},'86.3 ± 1.5','skip',None,'log parse failed and writeup parse failed',t0); print_result(r); raise SystemExit
    mean,sd=vals['id_e2s_mean'],vals['id_e2s_sd']
ok=abs(mean-86.3)<=0.5 and abs(sd-1.5)<=0.3
r=finish(item,{'source_used':source,'mean':mean,'sd':sd,'values_from_logs':log_vals},'86.3 ± 1.5','pass' if ok else 'fail',ok,None if ok else 'mean/sd outside tolerance',t0)
print_result(r)
