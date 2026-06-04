from common_v4 import *
import pandas as pd


t0 = start(); item = 'SA-3'
obs = {}
try:
    p1 = ROOT / 'data/esfm/positive_train_val_seq_smi.pt'
    p2 = ROOT / 'data/esfm/positive_test_seq_smi.pt'

    h1 = p1.read_bytes()[:15]
    h2 = p2.read_bytes()[:15]
    pointer = h1.startswith(b'version http') or h2.startswith(b'version http')
    obs['lfs_pointer_detected'] = pointer

    if pointer:
        import subprocess
        subprocess.run([
            'git', 'lfs', 'pull', '--include=data/esfm/positive_*.pt'
        ], cwd=ROOT, check=False)

    h1_after = p1.read_bytes()[:15]
    h2_after = p2.read_bytes()[:15]
    obs['lfs_pointer_after_pull'] = h1_after.startswith(b'version http') or h2_after.startswith(b'version http')
    if obs['lfs_pointer_after_pull']:
        raise RuntimeError('PT files still appear to be LFS pointers after git lfs pull')

    import torch, sys
    sys.path.insert(0, str(ROOT / 'src'))
    from calm.preprocess.esfm import extract_main_substrate

    tv = torch.load(p1, weights_only=False)
    te = torch.load(p2, weights_only=False)
    all_tuples = list(tv.values()) + list(te.values())

    raw = len(all_tuples)
    skipped = 0
    surviving = []
    for substrate_raw, enzyme_seq in all_tuples:
        m = extract_main_substrate(substrate_raw)
        if m is None:
            skipped += 1
        else:
            surviving.append((enzyme_seq, m))

    after = len(surviving)
    df = pd.DataFrame(surviving, columns=['enzyme_seq', 'substrate_smiles'])
    before = len(df)
    dd = df.drop_duplicates(['enzyme_seq', 'substrate_smiles'])
    ded = before - len(dd)
    final = len(dd)

    obs.update({
        'SA-3a_raw': raw,
        'SA-3b_cofactor_skipped': skipped,
        'SA-3c_after_cofactor': after,
        'SA-3d_dedup_dropped': ded,
        'final': final,
        'arithmetic_closes': (raw - skipped - ded) == final,
        'final_enzymes': int(dd.enzyme_seq.nunique()),
        'final_substrates': int(dd.substrate_smiles.nunique()),
    })

    exp = {
        'SA-3a_raw': 178463,
        'SA-3b_cofactor_skipped': 938,
        'SA-3c_after_cofactor': 177525,
        'SA-3d_dedup_dropped': 83,
        'final': 177442,
        'arithmetic_closes': True,
        'final_enzymes': 177389,
        'final_substrates': 2646,
    }

    ok = all(obs[k] == v for k, v in exp.items())
    r = finish(item, obs, exp, 'pass' if ok else 'fail', ok, None if ok else 'mismatch in one or more SA-3 values', t0)
except Exception as e:
    r = finish(item, {**obs, 'error': str(e)}, 'exact expected SA-3 values', 'fail', False, 'SA-3 execution failed', t0)

print_result(r)
