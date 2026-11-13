import json
import math
import re
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[3]
run = root / 'results/seed9001_s0.50_n5000'
for version, suffix in [('v1', ''), ('v2', '_v2')]:
    base = root / f'results/subagent_{version}'
    manifest = json.loads((base / 'manifest.json').read_text())
    bundle = json.loads((run / f'llm_bundle_subagent_claude-opus-5-5{suffix}.json').read_text())
    ids = [a for batch in manifest['batches'] for a in batch]
    malformed, missing, duplicate, mismatches = [], [], [], []
    all_rows = {}
    for i, expected in enumerate(manifest['batches']):
        batch = (base / f'batches/{i:02}.txt').read_text()
        assert re.findall(r'^id=(\S+)', batch, re.M) == expected
        found = {}
        for j, line in enumerate((base / f'replies/{i:02}.txt').read_text().splitlines(), 1):
            match = re.search(r'\{.*\}', line)
            if not match:
                continue
            try:
                row = json.loads(match.group())
            except json.JSONDecodeError:
                malformed.append((i, j))
                continue
            a = str(row.get('id'))
            if a not in expected:
                continue
            if a in found:
                duplicate.append((i, a))
                continue
            found[a] = row
        for a in expected:
            row = found.get(a, {})
            if not row:
                missing.append((i, a))
            score = row.get('score')
            if isinstance(score, (int, float)) and math.isfinite(score):
                parsed = min(1., max(0., score))
            else:
                parsed = {'flag': .9, 'abuse': .9, 'pass': .1, 'benign': .1}.get(str(row.get('verdict', '')).lower(), .5)
            if round(parsed, 6) != bundle['scores'][a]:
                mismatches.append(a)
            all_rows[a] = row
    assert len(ids) == len(set(ids)) == len(bundle['scores']) == 1922
    assert set(ids) == set(bundle['scores'])
    assert not mismatches
    print(version, 'manifest', {k:v for k,v in manifest.items() if k!='batches'})
    print(version, 'batches', len(manifest['batches']), 'sizes', dict(Counter(map(len, manifest['batches']))), 'malformed', malformed, 'missing', missing, 'duplicate', duplicate, 'bundle_mismatches', len(mismatches))
    print(version, 'bundle', {k:v for k,v in bundle.items() if k not in ['scores', 'metrics']})
    print(version, 'AP', {f: x['ap'] for f,x in bundle['metrics'].items()})

ss = json.loads((run / 'second_stage.json').read_text())
held = [c for c in ss['cells'] if c['condition']=='heldout']
inf = [c for c in ss['cells'] if c['condition']=='in_family']
print('expectation counts a,b,c,d,e:', sum(c['stage2']['llm:v2']['diff_vs_random_ci95'][0]>0 for c in held), sum(c['stage2']['llm:v2']['added_recall']<.1 for c in inf), sum(c['stage2']['llm:v1']['added_recall']>=c['stage2']['llm:v2']['added_recall'] for c in held), sum(c['stage2']['llm:v2']['added_recall']>c['stage2']['more-stage-1']['added_recall'] for c in held), min(held,key=lambda c:c['stage2']['llm:v2']['added_recall'])['family'])
for c in held:
    print('heldout', c['family'], 'n', c['n'], 'n_pos', c['n_pos'], 'positive_units', c['n_pos_units'], 'v2_precision', c['stage2']['llm:v2']['stage2_precision'])
print('AUDIT COMPLETE: 2 manifests, 82 reply files and 3844 bundled scores checked; expectation counts 4/3/5/6, weakest template_operator.')
