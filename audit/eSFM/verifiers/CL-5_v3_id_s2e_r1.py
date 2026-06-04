from common_v3 import *

t0=start(); item='CL-5'
# try training logs first
log_vals=[]
for i in [0,1,2,3]:
    txt=(ROOT/f'logs/esfm/esfm_full_62858369_{i}.out').read_text()
    m=re.findall(r'Pool-512 S→E R@1\s*[:=]\s*([0-9.]+)',txt)
    if m: log_vals.append(float(m[-1]))
source='training_logs'
if len(log_vals)==4:
    mean=float(np.mean(log_vals)); sd=float(np.std(log_vals))
else:
    vals,found=parse_writeup_esfm_table(); source='writeup_§3.2'
    if not found: 
        r=finish(item,{'source':source},'61.8 ± 0.4','skip',None,'log parse failed and writeup parse failed',t0); print_result(r); raise SystemExit
    mean,sd=vals['id_s2e_mean'],vals['id_s2e_sd']
ok=abs(mean-61.8)<=0.3 and abs(sd-0.4)<=0.2
r=finish(item,{'source_used':source,'mean':mean,'sd':sd,'values_from_logs':log_vals},'61.8 ± 0.4','pass' if ok else 'fail',ok,None if ok else 'mean/sd outside tolerance',t0)
print_result(r)
