from common_v4 import *


t0 = start(); item = 'SA-8'
files = [
    'configs/model/encoder/esfm_ffn.yaml',
    'configs/train/encoder/esfm_full.yaml',
    'configs/data/db/esfm.yaml',
]

esm_needles = ['facebook/esm2_t33_650M_UR50D', 'esm2_t33_650M_UR50D']
mol_needles = ['ibm/MoLFormer-XL-both-10pct', 'MoLFormer-XL-both-10pct']

contents = {f: read_text(f) for f in files}

def first_match(needles):
    for fp in files:
        text = contents[fp]
        for n in needles:
            if n in text:  # exact substring match, not regex
                return fp, n
    return None, None

def hash_pinned_near(text):
    low = text.lower()
    return ('revision:' in low) or ('commit_hash:' in low)

esm_file, esm_match = first_match(esm_needles)
mol_file, mol_match = first_match(mol_needles)

esm_present = esm_file is not None
mol_present = mol_file is not None

esm_hash = hash_pinned_near(contents[esm_file]) if esm_file else False
mol_hash = hash_pinned_near(contents[mol_file]) if mol_file else False

both_present = esm_present and mol_present
warn_no_hash = both_present and not (esm_hash and mol_hash)

status = 'pass' if both_present else 'fail'
reason = None
if not both_present:
    reason = 'missing one or more required encoder identifier substrings'
elif warn_no_hash:
    reason = 'identifiers present; hash pinning not found (warn-but-pass per spec)'

observed = {
    'esm': {
        'identifier_present': esm_present,
        'file_containing': esm_file,
        'matched_substring': esm_match,
        'hash_pinned': esm_hash,
    },
    'molformer': {
        'identifier_present': mol_present,
        'file_containing': mol_file,
        'matched_substring': mol_match,
        'hash_pinned': mol_hash,
    },
    'warn_no_hash': warn_no_hash,
    'search_files': files,
}

r = finish(item, observed, 'both encoder identifiers present via exact substring match; hash pinning preferred', status, both_present if status != 'fail' else False, reason, t0)
print_result(r)
