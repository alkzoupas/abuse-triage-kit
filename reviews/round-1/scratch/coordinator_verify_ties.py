"""Coordinator check for finding 3: the counterexample, and whether stage-1 boundary ties occur in the real cells."""
import numpy as np
from triage_kit.eval.second_stage import two_stage, STAGE1
from triage_kit.detectors.feature_model import FeatureModel
from triage_kit.detectors.features import feature_table
from triage_kit.eval.run_eval import RunConfig
from triage_kit.eval.splits import HELDOUT_FOLDS, cell_ids, make_splits
from triage_kit.eval.truth import load_truth
from triage_kit.generator import FAMILIES
from triage_kit.schema import read_dataset

r = two_stage([1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1], 1)
print("counterexample added_recall (code):", round(r["added_recall"], 6), "| exact sequential expectation: 0.25")

cfg = RunConfig(seed=9001, s=0.5, n_accounts=5000, n_boot=0)
d = cfg.data_root / cfg.name
data = read_dataset(d); labels = load_truth(d).labels
splits = {sp.name: sp for sp in make_splits([ae.account for ae in data], labels)}
pos = {ae.account_id: i for i, ae in enumerate(data)}
test_ids = sorted(splits["in_family"].test_ids)
te = np.array([pos[a] for a in test_ids])
_, X, names = feature_table(data, reference=data)
stage1 = {}
for name, sp in splits.items():
    tr = np.array([pos[a] for a in sorted(sp.train_ids)])
    y = np.array([int(labels[a]["is_abuse"]) for a in sorted(sp.train_ids)])
    m = FeatureModel(STAGE1).fit_table(X[tr], y, names)
    stage1[name] = dict(zip(test_ids, m.predict_table(X[te]).tolist(), strict=True))
ho = {fam: f"heldout_{i}" for i, fold in enumerate(HELDOUT_FOLDS, 1) for fam in fold}
k = cfg.k
for cond in ("heldout", "in_family"):
    for fam in FAMILIES:
        sp = splits["in_family" if cond == "in_family" else ho[fam]]
        ids = cell_ids(sp, fam, labels)
        s = np.sort(np.array([stage1[sp.name][a] for a in ids]))[::-1]
        kth, nxt = s[k - 1], s[k]
        block = int(np.sum(s == kth))
        above = int(np.sum(s > kth))
        print(f"{cond:9} {fam:18} k-th={kth:.6f} next={nxt:.6f} tie_at_boundary={kth == nxt} block_size={block} taken_from_block={k - above}")
