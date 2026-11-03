import triage_kit


def test_package_has_version() -> None:
    assert triage_kit.__version__ == "0.1.0"


def test_confirm_plan_uses_only_fresh_declared_seeds() -> None:
    from triage_kit.confirm import plan
    from triage_kit.demo import ROOT
    from triage_kit.report import load_prereg

    prereg = load_prereg(ROOT / "PREREGISTRATION.md")
    p = plan(prereg)
    used = {prereg["primary_seed"], *prereg["replication_seeds"], 101, 102, 103}
    assert {sd for sd, _, _ in p} == {9004, 9005, 9006} and not used & {sd for sd, _, _ in p}
    assert sorted({(sv, m) for _, sv, m in p}) == [
        (0.0, "standard"),
        (0.5, "ordinary"),
        (0.5, "standard"),
    ]


def test_confirm_large_plan_uses_fresh_seeds_at_20k() -> None:
    from triage_kit.confirm import plan, plan_large
    from triage_kit.demo import ROOT
    from triage_kit.report import load_prereg

    prereg = load_prereg(ROOT / "PREREGISTRATION.md")
    p = plan_large(prereg)
    earlier = {sd for sd, _, _ in plan(prereg)} | {
        prereg["primary_seed"],
        *prereg["replication_seeds"],
    }
    assert {sd for sd, _, _ in p} == {9007, 9008, 9009} and not earlier & {sd for sd, _, _ in p}
    assert prereg["confirmation_large"]["n_accounts"] == 20000
    assert sorted({(sv, m) for _, sv, m in p}) == [(0.0, "standard"), (0.5, "ordinary")]
