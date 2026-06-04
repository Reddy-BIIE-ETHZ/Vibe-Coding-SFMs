from common_v4 import *
import yaml


t0 = start(); item = 'SA-9'
try:
    text = read_text('src/calm/encoder/data.py')
    lines = text.splitlines()

    # confirm expected source-location semantics
    l310 = lines[309] if len(lines) >= 310 else ''
    l320 = lines[319] if len(lines) >= 320 else ''
    l332 = lines[331] if len(lines) >= 332 else ''

    system_random_used = 'SystemRandom' in text
    locations_ok = (
        'SystemRandom' in l310 and
        'self.rng.shuffle(self.by_cluster[c])' in l320 and
        'self.rng.shuffle(clusters)' in l332
    )

    spec = yaml.safe_load(read_text('audit/esfm_audit_v0.4.yaml'))
    rerun = float(spec['part_a_split_audit']['fold0_rerun_value'])
    original = float(spec['part_a_split_audit']['fold0_original_value'])
    diff = abs(rerun - original)
    within = diff < 0.8

    status = 'pass' if (system_random_used and locations_ok and within) else 'fail'
    reason = None if status == 'pass' else 'SystemRandom locations mismatch or rerun variance bound exceeded'

    observed = {
        'system_random_used': system_random_used,
        'source_locations': [310, 320, 332],
        'line_310': l310.strip(),
        'line_320': l320.strip(),
        'line_332': l332.strip(),
        'locations_match_expected': locations_ok,
        'fold0_original': original,
        'fold0_rerun': rerun,
        'rerun_minus_original': diff,
        'within_fold_sd': within,
    }

    r = finish(item, observed, 'SystemRandom confirmed at lines 310/320/332 and abs(rerun-original) < 0.8', status, within if status == 'pass' else False, reason, t0)
except Exception as e:
    r = finish(item, {'error': str(e)}, 'SA-9 variance check', 'fail', False, 'SA-9 execution failed', t0)

print_result(r)
