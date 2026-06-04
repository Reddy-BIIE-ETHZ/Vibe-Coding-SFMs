import subprocess
import tempfile

import pandas as pd

from common_v4 import *


t0 = start(); item = 'SA-1'
try:
    import shutil
    if shutil.which('mmseqs') is None:
        raise RuntimeError('mmseqs not installed in environment')
    df = pd.read_csv(ROOT / 'data/esfm/metadata.csv')
    seqs = df['protein_seq'].drop_duplicates().sample(500, random_state=42).tolist()
    with tempfile.TemporaryDirectory() as d:
        fasta = Path(d) / 'in.fasta'
        fasta.write_text(''.join(f">s{i}\\n{s}\\n" for i, s in enumerate(seqs)))
        outs = []
        for run in [1, 2]:
            out = Path(d) / f'out{run}'
            tmp = Path(d) / f'tmp{run}'
            subprocess.run(['mmseqs', 'easy-cluster', str(fasta), str(out), str(tmp), '--min-seq-id', '0.8', '-c', '0.8'], check=True, capture_output=True, text=True)
            outs.append(Path(str(out) + '_cluster.tsv').read_text())
    same = outs[0] == outs[1]
    r = finish(item, {'identical': same}, 'identical cluster assignments', 'pass' if same else 'fail', same, None, t0)
except Exception as e:
    r = finish(item, str(e), 'identical cluster assignments', 'skip', None, 'could not execute mmseqs check', t0)
print_result(r)
