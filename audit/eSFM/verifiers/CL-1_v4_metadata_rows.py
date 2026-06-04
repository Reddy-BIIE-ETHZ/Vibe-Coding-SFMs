from common_v4 import *
import pandas as pd

t0=start(); item='CL-1'
rows=len(pd.read_csv(ROOT/'data/esfm/metadata.csv'))
ok=rows==177442
r=finish(item,rows,177442,'pass' if ok else 'fail',ok,None if ok else 'row count mismatch',t0)
print_result(r)
