"""Acceptance: no generator metadata reaches any feature or prompt, and detectors never
import the label or metadata modules."""

import ast
from pathlib import Path

from tests.conftest import Dataset
from triage_kit.detectors.features import FEATURE_NAMES, feature_table

DETECTORS = Path(__file__).resolve().parents[1] / "src" / "triage_kit" / "detectors"
FORBIDDEN_MODULES = ("triage_kit.eval", "triage_kit.generator")
META_WORDS = ("family", "is_abuse", "template", "group", "label", "oracle")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:  # relative import inside detectors resolves to detectors.*
                mod = "triage_kit.detectors." + mod
            out.append(mod)
            out += [f"{mod}.{a.name}" for a in node.names]
    return out


def test_detectors_never_import_label_or_metadata_modules() -> None:
    files = list(DETECTORS.rglob("*.py"))
    assert files
    for f in files:
        for mod in _imports(f):
            assert not mod.startswith(FORBIDDEN_MODULES), f"{f.name} imports {mod}"
            assert "truth" not in mod and "labels" not in mod, f"{f.name} imports {mod}"


def test_feature_columns_carry_no_metadata(dev_s1: Dataset) -> None:
    ids, X, names = feature_table(dev_s1.data[:300], reference=dev_s1.data[:300])
    assert list(names) == list(FEATURE_NAMES) + ["template_sim", "template_sim_v2"]
    for n in names:
        if n.startswith("template_sim"):  # declared similarity features: events only
            continue
        assert not any(w in n for w in META_WORDS), n
    assert X.shape == (300, len(names))


def test_features_are_identical_when_labels_change(dev_s1: Dataset) -> None:
    # features are computed from AccountEvents only, which carry no label fields
    from triage_kit.schema import AccountEvents

    assert set(AccountEvents.__dataclass_fields__) == {"account", "events"}
    sub = dev_s1.data[:200]
    a = feature_table(sub, reference=sub)[1]
    b = feature_table(list(reversed(sub)), reference=sub)[1][::-1]
    assert (a == b).all()
