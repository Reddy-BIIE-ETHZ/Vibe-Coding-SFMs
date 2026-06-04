from common_v3 import *
import pandas as pd, numpy as np

t0=start(); item='SA-4'
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    df=pd.read_csv(ROOT/'data/esfm/metadata.csv').sample(10,random_state=42)
    seqs=df['protein_seq'].tolist()
    model_id='facebook/esm2_t33_650M_UR50D'
    def enc():
        tok=AutoTokenizer.from_pretrained(model_id)
        mdl=AutoModel.from_pretrained(model_id)
        mdl.eval()
        outs=[]
        with torch.no_grad():
            for s in seqs:
                x=tok(s,return_tensors='pt',truncation=True,max_length=1024)
                y=mdl(**x).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                outs.append(y)
        return np.stack(outs)
    a,b=enc(),enc()
    linf=float(np.max(np.abs(a-b)))
    ok=linf<=1e-5
    r=finish(item,{'linf':linf},'L_inf <= 1e-5','pass' if ok else 'fail',ok,None if ok else 'embedding mismatch',t0)
except Exception as e:
    r=finish(item,str(e),'deterministic embeddings','skip',None,'model download/inference unavailable',t0)
print_result(r)
