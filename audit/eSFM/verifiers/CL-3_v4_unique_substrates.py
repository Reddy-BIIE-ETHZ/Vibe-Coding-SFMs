from common_v4 import *
import pandas as pd

t0=start(); item='CL-3'
v=int(pd.read_csv(ROOT/'data/esfm/metadata.csv')['substrate_smiles'].nunique())
ok=v==2646
r=finish(item,v,2646,'pass' if ok else 'fail',ok,None if ok else 'unique substrate_smiles mismatch',t0)
print_result(r)
