from common_v4 import *

t0=start(); item='SA-7'
text=read_text('scripts/euler_mmseqs_esfm.slurm')
ok=('SEQ_COL="protein_seq"' in text and 'THRESHOLDS="0.4 0.6 0.8"' in text)
r=finish(item,{'seq_col_protein_seq':'SEQ_COL="protein_seq"' in text,'thresholds_040_060_080':'0.4 0.6 0.8' in text},'protein_seq at thresholds 0.4/0.6/0.8','pass' if ok else 'fail',ok,None if ok else 'missing seq_col or thresholds',t0)
print_result(r)
