import re

from common_v3 import *


t0 = start(); item = 'SA-8'
t1 = read_text('configs/train/encoder/esfm_full.yaml')
t2 = read_text('configs/model/encoder/esfm_ffn.yaml')
combined = t1 + '\n' + t2
esm = ('esm2_t33_650M_UR50D' in combined or 'facebook/esm2_t33_650M_UR50D' in combined)
mol = ('MoLFormer-XL' in combined or 'ibm/MoLFormer-XL-both-10pct' in combined)
hash_fields = bool(re.search(r'(revision|commit_hash|hf_hash)\s*:\s*\S+', combined))
if esm and mol and hash_fields:
    status = 'pass'; reason = None
elif esm and mol:
    status = 'pass'; reason = 'identifiers present but no commit-hash pinning'
else:
    status = 'fail'; reason = 'missing one or more encoder identifiers'
r = finish(item, {'esm_identifier_present': esm, 'molformer_identifier_present': mol, 'hash_pinned': hash_fields}, 'identifiers required; hash pinning preferred', status, (status != 'fail'), reason, t0)
print_result(r)
