from common_v3 import *
import pandas as pd

t0=start(); item='CL-2'
v=int(pd.read_csv(ROOT/'data/esfm/metadata.csv')['protein_seq'].nunique())
ok=v==177389
r=finish(item,v,177389,'pass' if ok else 'fail',ok,None if ok else 'unique protein_seq mismatch',t0)
print_result(r)
