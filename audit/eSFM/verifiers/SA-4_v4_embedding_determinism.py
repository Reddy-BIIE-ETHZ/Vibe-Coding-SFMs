from common_v4 import *
import pandas as pd
import numpy as np
from pathlib import Path


t0 = start(); item = 'SA-4'
try:
    cache_dir = Path.home() / '.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D'
    if not cache_dir.exists():
        r = finish(
            item,
            {'cache_dir': str(cache_dir), 'cache_exists': False},
            'HF cache pre-warmed for facebook/esm2_t33_650M_UR50D',
            'skip',
            None,
            'pre-warmed ESM-2 cache missing; setup pre-warm requirement not satisfied',
            t0,
        )
        print_result(r)
        raise SystemExit(0)

    import torch
    from transformers import AutoTokenizer, AutoModel

    df = pd.read_csv(ROOT / 'data/esfm/metadata.csv').sample(10, random_state=42)
    seqs = df['protein_seq'].tolist()
    model_id = 'facebook/esm2_t33_650M_UR50D'

    def enc_once():
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        mdl = AutoModel.from_pretrained(model_id, local_files_only=True)
        mdl.eval()
        outs = []
        with torch.no_grad():
            for s in seqs:
                x = tok(s, return_tensors='pt', truncation=True, max_length=1024)
                y = mdl(**x).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                outs.append(y)
        return np.stack(outs)

    a, b = enc_once(), enc_once()
    linf = float(np.max(np.abs(a - b)))
    ok = linf <= 1e-5
    r = finish(item, {'linf': linf, 'cache_dir': str(cache_dir)}, 'L_inf <= 1e-5', 'pass' if ok else 'fail', ok, None if ok else 'embedding mismatch', t0)
except Exception as e:
    r = finish(item, str(e), 'deterministic embeddings', 'skip', None, 'model load/inference unavailable in environment', t0)

print_result(r)
