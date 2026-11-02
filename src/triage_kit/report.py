"""Build the static report (report/index.html) from results/ metrics files.

No server, no JS: inline CSS, inline SVG charts (hover text via SVG <title>).
Every LLM number from the mock is labeled "MOCK — not a result".
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from triage_kit.eval.splits import HELDOUT_FOLDS
from triage_kit.generator import FAMILIES

MOCK_TAG = "MOCK — not a result"
DETS_SHOWN = ("rules", "lr", "lr+sim", "gb", "gb+sim", "llm:mock")
REQUIRED_SECTIONS = (
    "Setup",
    "Null control",
    "In-family vs held-out",
    "Template operator: with and without the similarity feature",
    "Cost",
    "Expected vs actual (pre-registration)",
    "What this can't tell you",
)

CSS = """
:root{--surface:#fcfcfb;--panel:#f4f3f0;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--grid:#e4e2dc;--s1:#2a78d6;--s2:#eb6834;--hit:#1f7a1f;--miss:#b3261e;--warnbg:#fff4d6;}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){--surface:#1a1a19;
--panel:#242422;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--grid:#3a3936;--s1:#3987e5;
--s2:#d95926;--hit:#6cc46c;--miss:#ff8a80;--warnbg:#3a3220;}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#242422;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e86;--grid:#3a3936;--s1:#3987e5;--s2:#d95926;--hit:#6cc46c;--miss:#ff8a80;--warnbg:#3a3220;}
body{background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,sans-serif;margin:0;}
main{max-width:1040px;margin:0 auto;padding:16px;}
h1{font-size:26px;margin:8px 0} h2{margin-top:40px;border-bottom:1px solid var(--grid);padding-bottom:4px}
h3{margin-top:24px} p,li{color:var(--ink)} .sub{color:var(--ink2)}
table{border-collapse:collapse;margin:8px 0 16px;font-size:13px;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--grid);padding:4px 8px;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600} td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.mock{color:var(--muted)} .mock em{font-style:normal;font-size:10px;border:1px solid var(--muted);
border-radius:3px;padding:0 3px;margin-left:3px;white-space:nowrap}
.banner{background:var(--warnbg);border-radius:6px;padding:10px 14px;margin:12px 0}
.hit{color:var(--hit);font-weight:600} .miss{color:var(--miss);font-weight:600}
.flag{color:var(--miss);font-size:12px} svg{max-width:100%;height:auto} svg text{fill:var(--ink2)}
.legend span{display:inline-block;margin-right:16px;color:var(--ink2);font-size:13px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
code{background:var(--panel);padding:0 3px;border-radius:3px}
"""


def load_prereg(path: Path) -> dict[str, Any]:
    m = re.search(r"```prereg\n(.*?)```", path.read_text(), flags=re.DOTALL)
    if not m:
        raise ValueError(f"{path}: no ```prereg block")
    out: dict[str, Any] = yaml.safe_load(m.group(1))
    return out


def load_run(results_root: Path, seed: int, s: float, n: int = 5000) -> dict[str, Any]:
    return json.loads((results_root / f"seed{seed}_s{s:.2f}_n{n}" / "metrics.json").read_text())


def heldout_split_of(family: str) -> str:
    for k, fold in enumerate(HELDOUT_FOLDS, 1):
        if family in fold:
            return f"heldout_{k}"
    raise KeyError(family)


def cell(run: dict[str, Any], split: str, family: str, det: str) -> dict[str, Any]:
    for c in run["cells"]:
        if c["split"] == split and c["family"] == family and c["detector"] == det:
            return c
    raise KeyError((split, family, det))


def esc(x: Any) -> str:
    return html.escape(str(x))


def f3(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def num(x: float | None, det: str = "", ci: Sequence[float | None] | None = None) -> str:
    txt = f3(x) + (f" [{f3(ci[0])}, {f3(ci[1])}]" if ci else "")
    if det.startswith("llm:mock"):
        return f'<span class="mock">{txt}<em>{MOCK_TAG}</em></span>'
    return txt


def ci_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def winner(
    run: dict[str, Any], split: str, family: str, primary: str, key: str = "ap_ci95"
) -> tuple[str, bool, str]:
    r = cell(run, split, family, "rules")
    f = cell(run, split, family, primary)
    decisive = not ci_overlap(r.get(key, r["ap_ci95"]), f.get(key, f["ap_ci95"]))
    if not decisive:
        return "tie", False, f"rules {f3(r['ap'])} vs {primary} {f3(f['ap'])}: 95% CIs overlap"
    w = "rules" if r["ap"] > f["ap"] else "feature_model"
    return w, True, f"rules {f3(r['ap'])} vs {primary} {f3(f['ap'])}: CIs do not overlap"


# ------------------------------------------------------------------ charts
def dot_whisker(run: dict[str, Any], family: str) -> str:
    """Rows = detectors; two marks per row (in-family, held-out); x = PR-AUC 0..1."""
    W, left, right, top, row_h = 640, 170, 20, 26, 26
    dets = DETS_SHOWN
    H = top + row_h * len(dets) + 34
    x = lambda v: left + (W - left - right) * v  # noqa: E731
    ho = heldout_split_of(family)
    base = cell(run, "in_family", family, "rules")["base_rate"]
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="PR-AUC for {esc(family)}">',
    ]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{H - 28}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{x(t):.1f}" y="{H - 12}" font-size="11" text-anchor="middle">{t:g}</text>'
        )
    parts.append(
        f'<line x1="{x(base):.1f}" x2="{x(base):.1f}" y1="{top - 6}" y2="{H - 28}" '
        f'stroke="var(--ink2)" stroke-dasharray="3 3"><title>base rate {base:.3f}</title></line>'
        f'<text x="{x(base) + 4:.1f}" y="{top - 10}" font-size="10">base rate {base:.3f}</text>'
    )
    for i, det in enumerate(dets):
        cy = top + row_h * i + row_h / 2
        label = det + (" (MOCK)" if det.startswith("llm") else "")
        parts.append(
            f'<text x="{left - 8}" y="{cy + 4:.1f}" font-size="12" text-anchor="end">{esc(label)}</text>'
        )
        for j, (split, color, filled) in enumerate(
            (("in_family", "var(--s1)", True), (ho, "var(--s2)", False))
        ):
            c = cell(run, split, family, det)
            yy = cy - 5 + 10 * j
            lo, hi = c["ap_ci95"]
            mock = det.startswith("llm")
            stroke = "var(--muted)" if mock else color
            tip = (
                f"{det} {'in-family' if j == 0 else 'held-out'}: PR-AUC {f3(c['ap'])} "
                f"[{f3(lo)}, {f3(hi)}], base {f3(c['base_rate'])}"
                + (f" — {MOCK_TAG}" if mock else "")
            )
            fill = stroke if filled else "var(--surface)"
            parts.append(
                f"<g><title>{esc(tip)}</title>"
                f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                f'stroke="{stroke}" stroke-width="2" stroke-linecap="round"/>'
                f'<circle cx="{x(c["ap"]):.1f}" cy="{yy:.1f}" r="4.5" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="2"/></g>'
            )
    parts.append("</svg>")
    return "".join(parts)


LEGEND = (
    '<div class="legend"><span><i class="dot" style="background:var(--s1)"></i>in-family '
    '(trained on all six families)</span><span><i class="dot" style="border:2px solid '
    'var(--s2);width:7px;height:7px"></i>held-out (this family unseen in training)</span>'
    "<span>whiskers = 95% bootstrap CI; dashed = base rate</span></div>"
)


# ------------------------------------------------------------------ sections
def _setup(prereg: dict[str, Any], primary: dict[str, Any], seeds: dict[str, Any]) -> str:
    rows = []
    for name, fam in FAMILIES.items():
        n_test = cell(primary, "in_family", name, "rules")["n_pos"]
        groups = primary["test_groups"].get(name)
        g = f"{groups} groups" if groups is not None else "—"
        rows.append(
            f"<tr><td><b>{esc(name)}</b></td><td>{esc(fam.mechanism)}</td>"
            f"<td>{esc(', '.join(fam.channels))}</td><td class=n>{n_test}</td><td class=n>{g}</td></tr>"
        )
    cfg = primary["config"]
    return f"""
<h2>Setup</h2>
<p>Synthetic marketplace: {cfg["n_accounts"]:,} accounts over 60 days, a benign population plus six
planted abuse families. A signal-strength knob <code>s</code> scales every mechanism; at
<code>s = 0</code> abuse accounts are drawn from the benign distribution. Primary analysis:
<code>s = {prereg["primary_s"]}</code>, eval seed {prereg["primary_seed"]} (bootstrap CIs);
replication seeds {", ".join(map(str, prereg["replication_seeds"]))} (point estimates). Dev seeds
{", ".join(map(str, prereg["dev_seeds"]))} were used for building and CI and never appear here.</p>
<p>Split: earliest 60% of units train, the rest test ({primary["n_test"]:,} test accounts). A unit is
an account, or a whole group for coordinated families, so no account or group straddles the split.
<b>In-family</b> trains on all six families; <b>held-out</b> trains on four and tests on the two unseen
ones (three rotations). Each cell's test pool is benign test accounts plus one family.</p>
<p>Detectors: <b>rules</b> (<code>rules.yaml</code>, eleven readable rules; <b>the rules encode human
priors</b> written from the mechanism descriptions below, by someone who knew them; in a split only
rules tagged with a seen family are active); <b>feature model</b> (logistic regression <code>lr</code>
and gradient boosting <code>gb</code>, each with and without <code>+sim</code>, the template-similarity
feature); <b>LLM reviewer</b> (here the offline <code>MockLLM</code>, whose score is a hash of the
prompt: {MOCK_TAG}). Metrics: PR-AUC with base rate, precision/recall at a review budget of
k = {prereg["review_budget_k"]}, {prereg["bootstrap_samples"]:,}-sample stratified cluster bootstrap
95% CIs (resampled by account, by group for coordinated families).</p>
<p class="banner"><b>Disclosure.</b> The eval seeds have been run three times. (1) The first
<code>make demo</code> failed the null control (4 of 84 cells); the cause, a split artifact that made
test group members older than benign test accounts at s = 0, was confirmed on dev seeds and fixed in
the split. (2) The second run passed and was published. (3) After reading it, the owner asked for
better confidence intervals (few-cluster widening, a tie-aware null PR-AUC) and a better similarity
feature (v2). All three were designed on dev seeds and declared, with dates, in PREREGISTRATION.md
(AMENDMENTS) before this third run. No generator, detector rule or pre-registered model setting
changed. Post-hoc items are labelled where they appear.</p>
<table><tr><th>family</th><th>mechanism (as written before any detector)</th><th>evidence
channel</th><th>abuse test accounts</th><th>test groups</th></tr>{"".join(rows)}</table>
"""


def _null(null: dict[str, Any]) -> str:
    cells = null["cells"]
    bad = [
        c
        for c in cells
        if not (c["ap_ci_bonf"][0] <= c["base_rate"] <= c["ap_ci_bonf"][1])
        or not (c["roc_ci_bonf"][0] <= 0.5 <= c["roc_ci_bonf"][1])
    ]
    unc = sum(not (c["ap_ci95"][0] <= c["base_rate"] <= c["ap_ci95"][1]) for c in cells)
    rows = []
    for det in (*DETS_SHOWN, "oracle"):
        cs = [c for c in cells if c["detector"] == det]
        worst = max(cs, key=lambda c: abs(c["ap"] - c["base_rate"]))
        rows.append(
            f"<tr><td>{esc(det)}</td><td class=n>{len(cs)}</td>"
            f"<td class=n>{num(worst['ap'] - worst['base_rate'], det)}</td>"
            f"<td class=n>{num(min(c['roc'] for c in cs), det)} – {num(max(c['roc'] for c in cs), det)}</td>"
            f"<td class=n>{sum(c in bad for c in cs)}</td></tr>"
        )
    verdict = (
        '<p class="hit">Pass: no cell\'s corrected interval excludes the base rate (PR-AUC) or 0.5 (ROC-AUC).</p>'
        if not bad
        else f'<p class="miss">FAIL: {len(bad)} cells exclude the null. Treat every result below as suspect.</p>'
    )
    return f"""
<h2>Null control</h2>
<p>At <code>s = 0</code> there is nothing to find. All {null["n_cells"]} detector × family × split cells
(oracle included) are treated as one family of tests with a Bonferroni-corrected interval
(alpha = 0.05 / {null["n_cells"]} = {null["alpha_bonferroni"]:.5f}).</p>
{verdict}
<p class="sub">For scale: {unc} of {len(cells)} <i>uncorrected</i> 95% PR-AUC intervals exclude the base
rate, versus about {len(cells) * 0.05:.0f} expected by chance if cells were independent (they are
not: cells share test accounts). This is why the correction is not optional.</p>
<p class="sub">Two post-hoc refinements (dated in PREREGISTRATION.md, AMENDMENTS): every interval is
widened for the number of resampled units in the scarcer class (a t-based expansion that matters
most for coordinated families with ~10 groups), and each PR-AUC is also compared with its exact
<b>null PR-AUC</b>, the expected PR-AUC if labels were assigned at random given that detector's
ties. For a continuous score with few positives the null PR-AUC sits above the base rate.
{sum(c.get("degenerate", False) for c in cells)} cells are <b>degenerate</b> (no abuse account scored
above the lowest score, so PR-AUC equals the base rate and cannot vary); they are no-signal cells by
construction. Non-degenerate cells whose corrected interval excludes the null PR-AUC:
{sum((not c.get("degenerate", False)) and not (c["ap_ci_bonf"][0] <= c.get("ap_random", c["base_rate"]) <= c["ap_ci_bonf"][1]) for c in cells)}.</p>
<table><tr><th>detector</th><th>cells</th><th>largest |PR-AUC − base rate|</th><th>ROC-AUC range</th>
<th>cells failing</th></tr>{"".join(rows)}</table>
"""


def _signal(s1: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{esc(f)}</td><td class=n>{f3(cell(s1, 'in_family', f, 'oracle')['ap'])}</td>"
        f"<td class=n>{f3(cell(s1, 'in_family', f, 'oracle')['base_rate'])}</td></tr>"
        for f in FAMILIES
    )
    return f"""
<h2>Signal check (oracle)</h2>
<p>The oracle knows each planted mechanism (it reads generator internals and is not a detector). At
<code>s = 1</code> it shows the signal is present in the events. Required: PR-AUC ≥ 0.9 per family.</p>
<table><tr><th>family</th><th>oracle PR-AUC (s = 1)</th><th>base rate</th></tr>{rows}</table>
"""


def _family_table(run: dict[str, Any], few: int) -> str:
    out = []
    for fam in FAMILIES:
        ho = heldout_split_of(fam)
        rows = []
        for det in DETS_SHOWN:
            a, b = cell(run, "in_family", fam, det), cell(run, ho, fam, det)
            deg = lambda c: " (degenerate)" if c.get("degenerate") else ""  # noqa: E731
            rows.append(
                f"<tr><td>{esc(det)}</td><td class=n>{num(a['ap'], det, a['ap_ci95'])}{deg(a)}</td>"
                f"<td class=n>{num(a.get('ap_random'), det)}</td>"
                f"<td class=n>{num(b['ap'], det, b['ap_ci95'])}{deg(b)}</td>"
                f"<td class=n>{num(a['p_at_k'], det)} / {num(a['r_at_k'], det)}</td></tr>"
            )
        groups = run["test_groups"].get(fam)
        flag = (
            f'<p class="flag">Wide CIs because few groups: only {groups} {esc(fam)} groups reached the '
            f"test split (threshold {few}). Group-level resampling makes these intervals wide; read "
            "differences between detectors here with caution.</p>"
            if groups is not None and groups < few
            else ""
        )
        base = cell(run, "in_family", fam, "rules")["base_rate"]
        out.append(
            f"<h3>{esc(fam)}</h3>{flag}{LEGEND}{dot_whisker(run, fam)}"
            f"<table><tr><th>detector</th><th>PR-AUC in-family [95% CI]</th><th>null PR-AUC (no signal)</th><th>PR-AUC held-out "
            f"({esc(ho)}) [95% CI]</th><th>in-family precision / recall @k</th></tr>{''.join(rows)}"
            f"</table><p class=sub>base rate {base:.3f}</p>"
        )
    return "".join(out)


def _in_vs_held(primary: dict[str, Any], secondary: dict[str, Any], prereg: dict[str, Any]) -> str:
    few = prereg["few_groups_threshold"]
    return f"""
<h2>In-family vs held-out</h2>
<p>Primary analysis, <code>s = {primary["config"]["s"]}</code>, seed {primary["config"]["seed"]}.
In-family: the family was in training. Held-out: it was not, and the rules written for it are
switched off. Hover a mark for its numbers; each chart has a table view below it. A whisker that
reaches the <b>null PR-AUC</b> column means "no detectable signal" for that detector.</p>
{_family_table(primary, few)}
<h3>Secondary: full signal strength (s = {secondary["config"]["s"]})</h3>
<table><tr><th>family</th>{"".join(f"<th>{esc(d)} in / held</th>" for d in DETS_SHOWN)}</tr>
{
        "".join(
            "<tr><td>"
            + esc(f)
            + "</td>"
            + "".join(
                f"<td class=n>{num(cell(secondary, 'in_family', f, d)['ap'], d)} / "
                f"{num(cell(secondary, heldout_split_of(f), f, d)['ap'], d)}</td>"
                for d in DETS_SHOWN
            )
            + "</tr>"
            for f in FAMILIES
        )
    }</table>
"""


def _template(runs: dict[float, dict[str, Any]]) -> str:
    dets = ("lr", "lr+sim", "lr+sim2", "gb", "gb+sim", "gb+sim2")
    rows = []
    for s, run in sorted(runs.items()):
        for split in ("in_family", heldout_split_of("template_operator")):
            vals = {d: cell(run, split, "template_operator", d) for d in dets}
            rows.append(
                f"<tr><td class=n>{s:g}</td><td>{esc(split)}</td>"
                + "".join(
                    f"<td class=n>{num(v['ap'], d, v['ap_ci95'])}</td>" for d, v in vals.items()
                )
                + f"<td class=n>{vals['gb+sim']['ap'] - vals['gb']['ap']:+.3f}</td>"
                + f"<td class=n>{vals['gb+sim2']['ap'] - vals['gb']['ap']:+.3f}</td></tr>"
            )
    return f"""
<h2>Template operator: with and without the similarity feature</h2>
<p>Template-operator accounts replay a shared script with small perturbations, so one account alone
looks ordinary; the evidence is that many accounts share near-identical action sequences. Both
similarity features <b>were designed knowing this mechanism</b>.</p>
<ul>
<li><code>template_sim</code> (<b>pre-registered</b>, the <code>+sim</code> variants): MinHash over
4-grams of event-type and log-gap tokens, max similarity to any other account. Declared before any
evaluation ran.</li>
<li><code>template_sim_v2</code> (<b>post-hoc, exploratory</b>, the <code>+sim2</code> variants):
added after v1 missed H4, at the owner's request, from a dev-seed diagnosis. Exact Jaccard over
type/gap n-grams plus listing co-visitation tokens, keeping only tokens shared by 2 accounts to 1%
of the population, scored as the mean of the 3 closest accounts. Declared in PREREGISTRATION.md
(AMENDMENTS) before the eval seeds were re-run, but it was designed after seeing v1 fail, so treat
its numbers as a lead to confirm on fresh data, not as a tested hypothesis.</li>
</ul>
<table><tr><th>s</th><th>split</th>{"".join(f"<th>{esc(d)}</th>" for d in dets)}
<th>gb+sim − gb</th><th>gb+sim2 − gb (post-hoc)</th></tr>{"".join(rows)}</table>
<p class=sub>Why v1 failed (found on a dev seed before any eval run, kept as declared): crew-mates'
exact shingle Jaccard is only about 0.02–0.09 (benign pairs ≤ 0.03), and 64 MinHash functions
(resolution ≈ 0.016) maximised over thousands of neighbours mostly measure noise.</p>
"""


def _budget(primary: dict[str, Any], k: int) -> str:
    rows = "".join(
        f"<tr><td>{esc(c['detector'])}</td><td class=n>{num(c['p_at_k'], c['detector'])}</td>"
        f"<td class=n>{num(c['r_at_k'], c['detector'])}</td>"
        f"<td class=n>{num(c['ap'], c['detector'], c.get('ap_ci95'))}</td></tr>"
        for c in primary["pooled"]
    )
    base = primary["pooled"][0]["base_rate"]
    return f"""
<h2>Review budget (all families together)</h2>
<p>If a team can review k = {k} accounts from the whole test population ({primary["n_test"]:,}
accounts, base rate {base:.3f}), how many are abuse? In-family split, s = {primary["config"]["s"]}.</p>
<table><tr><th>detector</th><th>precision @{k}</th><th>recall @{k}</th><th>PR-AUC [95% CI]</th></tr>{rows}</table>
"""


def _cost(primary: dict[str, Any], bundles: list[dict[str, Any]]) -> str:
    c = primary["cost"]
    ph = " (PLACEHOLDER prices from config/prices.yaml)" if c["llm_prices_placeholder"] else ""
    extra = "".join(
        f"<tr><td>real LLM: {esc(b['model'])} ({b['n_scored']} accounts)</td>"
        f"<td class=n>{b['tokens_in_total'] / b['n_scored']:.0f} / {b['tokens_out_total'] / b['n_scored']:.0f}</td>"
        f"<td class=n>see prices.yaml</td></tr>"
        for b in bundles
    )
    return f"""
<h2>Cost</h2>
<p>API cost per 1,000 accounts scored. Rules and the feature model need no API calls (their compute
cost is negligible at this scale and not measured here). The LLM line multiplies the mock's
<b>approximate</b> token counts (characters ÷ 4 of the real prompt the reviewer would send) by
placeholder prices{ph}. Replace the prices with your provider's before quoting any cost.</p>
<table><tr><th>detector</th><th>tokens in / out per account</th><th>cost per 1,000 accounts</th></tr>
<tr><td>rules</td><td class=n>0 / 0</td><td class=n>{c["rules_cost_per_1000"]:.2f}</td></tr>
<tr><td>feature model</td><td class=n>0 / 0</td><td class=n>{c["feature_model_cost_per_1000"]:.2f}</td></tr>
<tr><td>llm:mock priced as <code>{esc(c["llm_price_model"])}</code></td>
<td class=n>{num(c["llm_tokens_in_per_call"], "llm:mock")} / {num(c["llm_tokens_out_per_call"], "llm:mock")}</td>
<td class=n>{num(c["llm_cost_per_1000"], "llm:mock")} {esc(ph)}</td></tr>{extra}</table>
"""


def _expected(
    prereg: dict[str, Any],
    primary: dict[str, Any],
    s1: dict[str, Any],
    null: dict[str, Any],
    pf: str,
) -> str:
    rows = []
    hits = hits0 = misses = 0
    for kind in ("in_family", "heldout"):
        for fam in FAMILIES:
            split = "in_family" if kind == "in_family" else heldout_split_of(fam)
            exp = prereg["expected_winner"][kind][fam]
            act0, dec0, _ = winner(primary, split, fam, pf, "ap_ci95_plain")
            act, decisive, why = winner(primary, split, fam, pf)
            ok0 = act0 == exp
            ok = act == exp
            hits0 += ok0
            hits += ok
            misses += not ok
            rows.append(
                f"<tr><td>{esc(fam)}</td><td>{esc(kind)}</td><td>{esc(exp)}</td>"
                f"<td>{esc(act0)}{' (decisive)' if dec0 else ''} "
                f"<span class={'hit' if ok0 else 'miss'}>{'hit' if ok0 else 'MISS'}</span></td>"
                f"<td>{esc(act)}{' (decisive)' if decisive else ''}</td><td class={'hit' if ok else 'miss'}>"
                f"{'hit' if ok else 'MISS'}</td><td class=sub>{esc(why)}</td></tr>"
            )
    hyp = []
    null_ok = all(
        c["ap_ci_bonf"][0] <= c["base_rate"] <= c["ap_ci_bonf"][1]
        and c["roc_ci_bonf"][0] <= 0.5 <= c["roc_ci_bonf"][1]
        for c in null["cells"]
    )
    hyp.append(("H1 null control holds", null_ok, f"{null['n_cells']} cells, Bonferroni"))
    orc = [cell(s1, "in_family", f, "oracle")["ap"] for f in FAMILIES]
    hyp.append(
        ("H2 oracle PR-AUC ≥ 0.9 on every family at s = 1", min(orc) >= 0.9, f"min {min(orc):.3f}")
    )
    for fam in FAMILIES:
        a = cell(primary, "in_family", fam, pf)["ap"]
        b = cell(primary, heldout_split_of(fam), fam, pf)["ap"]
        hyp.append((f"H3 {fam}: {pf} lower held-out than in-family", b < a, f"{a:.3f} → {b:.3f}"))
    tf = "template_operator"
    g_in = (
        cell(primary, "in_family", tf, "gb+sim")["ap"] - cell(primary, "in_family", tf, "gb")["ap"]
    )
    ho = heldout_split_of(tf)
    g_ho = cell(primary, ho, tf, "gb+sim")["ap"] - cell(primary, ho, tf, "gb")["ap"]
    g2_in = (
        cell(primary, "in_family", tf, "gb+sim2")["ap"] - cell(primary, "in_family", tf, "gb")["ap"]
    )
    g2_ho = cell(primary, ho, tf, "gb+sim2")["ap"] - cell(primary, ho, tf, "gb")["ap"]
    need = prereg["template_sim_gain_in_family_min"]
    hyp.append((f"H4 +sim gain on {tf} in-family ≥ {need}", g_in >= need, f"gain {g_in:+.3f}"))
    hyp.append(
        ("H5 +sim gain smaller held-out than in-family", g_ho < g_in, f"{g_in:+.3f} vs {g_ho:+.3f}")
    )
    hrows = "".join(
        f"<tr><td>{esc(h)}</td><td class={'hit' if ok else 'miss'}>{'hit' if ok else 'MISS'}</td>"
        f"<td class=n>{esc(d)}</td></tr>"
        for h, ok, d in hyp
    )
    return f"""
<h2>Expected vs actual (pre-registration)</h2>
<p>Expectations were committed in PREREGISTRATION.md before any detector existed. Winner = higher
PR-AUC between rules and the primary feature model (<code>{esc(pf)}</code>); called decisive only if
the 95% CIs do not overlap, otherwise "tie". Misses are listed, not explained away.
The pre-registered interval method is shown first; the widened intervals (a post-hoc amendment)
second. Winner calls with pre-registered intervals: {hits0} hit, {len(rows) - hits0} miss; with widened
intervals: {hits} hit, {misses} miss.</p>
<table><tr><th>family</th><th>split</th><th>expected</th><th>actual (pre-registered CIs)</th>
<th>actual (widened CIs)</th><th></th><th>detail (widened)</th></tr>
{"".join(rows)}</table>
<table><tr><th>hypothesis</th><th></th><th>detail</th></tr>{hrows}
<tr><td>Exploratory (post-hoc, not a pre-registered hypothesis): +sim2 gain on {tf} in-family &gt; 0</td>
<td class={"hit" if g2_in > 0 else "miss"}>{"yes" if g2_in > 0 else "no"}</td><td class=n>gain {g2_in:+.3f} (held-out {g2_ho:+.3f})</td></tr>
<tr><td>H6 real LLM strongest on review_ring/ato, weakest on template_operator; costs far more</td>
<td>untestable offline</td><td>{MOCK_TAG}: the mock cannot test this</td></tr></table>
"""


def _replication(
    prereg: dict[str, Any], reps: list[dict[str, Any]], primary: dict[str, Any], pf: str
) -> str:
    runs = [primary, *reps]
    head = "".join(f"<th>seed {r['config']['seed']}</th>" for r in runs)
    rows = []
    for fam in FAMILIES:
        for det in ("rules", pf):
            for split in ("in_family", heldout_split_of(fam)):
                vals = "".join(
                    f"<td class=n>{f3(cell(r, split, fam, det)['ap'])}</td>" for r in runs
                )
                rows.append(
                    f"<tr><td>{esc(fam)}</td><td>{esc(det)}</td><td>{esc(split)}</td>{vals}</tr>"
                )
    return f"""
<h2>Replication</h2>
<p>PR-AUC point estimates at s = {prereg["primary_s"]} on the replication eval seeds.</p>
<table><tr><th>family</th><th>detector</th><th>split</th>{head}</tr>{"".join(rows)}</table>
"""


def _bundles(bundles: list[dict[str, Any]], primary_cells: dict[str, Any]) -> str:
    if not bundles:
        return (
            "<h2>Real LLM pass</h2><p>No real-model results bundle found. All LLM numbers in this "
            f"report come from the mock: {MOCK_TAG}. See the README to run a real pass.</p>"
        )
    out = ["<h2>Real LLM pass</h2>"]
    for b in bundles:
        out.append(
            f"<p>Model <code>{esc(b['model'])}</code>, {b['n_scored']} test accounts, prompt "
            f"{esc(b['prompt_version'])}. Metrics on the scored subset only:</p>"
            f"<table><tr><th>family</th><th>PR-AUC</th><th>base rate</th></tr>"
            + "".join(
                f"<tr><td>{esc(f)}</td><td class=n>{f3(v['ap'])}</td><td class=n>{f3(v['base_rate'])}</td></tr>"
                for f, v in b.get("_metrics", {}).items()
            )
            + "</table>"
        )
    return "".join(out)


CANT_TELL = """
<h2>What this can't tell you</h2>
<p>Written for someone deciding whether to run this kit on their own marketplace events.</p>
<ul>
<li><b>It can't tell you how well any detector catches <i>your</i> abuse.</b> Every number above comes
from synthetic accounts with planted, labelled patterns. Real abusers adapt, mix tactics and leave
no labels. A detector that wins here has only shown it can find these six cartoons.</li>
<li><b>On your own events you get rankings, not accuracy.</b> The bring-your-own-events adapter runs the
rules and feature model and tells you which accounts each would send to review first. Without
labels it cannot compute precision or recall. To get those, have investigators label a sample of
each detector's top-k (and a random sample, to estimate what is missed).</li>
<li><b>The feature model here was trained on synthetic labels.</b> Scoring your accounts with it is
out-of-distribution; treat its scores as a demonstration. Retrain on your own labelled history.</li>
<li><b>The rules are our priors, not your policy.</b> They were written by someone who knew exactly what
was planted, which is an advantage no real rule writer has. Use them as a readable starting set.</li>
<li><b>It says nothing about real LLM quality.</b> Every LLM number in this report is a placeholder from a
mock that hashes its prompt. Running the optional real-model pass gives you a measurement on this
synthetic data, which still says little about your data.</li>
<li><b>The costs are placeholders.</b> Token counts are approximate and prices come from a config file
with made-up values. Plug in your provider's price sheet and your own volumes.</li>
<li><b>What it can tell you:</b> whether your event export fits a simple, documented schema; what the
three approaches look like side by side on the same accounts; how much of a fixed review budget each
ranking would use up; and the shape of an honest comparison (pre-registered expectations, a null
control, held-out families and confidence intervals), which you can repeat once you have labels.</li>
</ul>
"""


def build_report(
    results_root: Path, prereg_path: Path, out_path: Path, commit: str = "unknown"
) -> Path:
    prereg = load_prereg(prereg_path)
    seed, s = prereg["primary_seed"], prereg["primary_s"]
    primary = load_run(results_root, seed, s)
    null = load_run(results_root, seed, 0.0)
    s1 = load_run(results_root, seed, 1.0)
    reps = [load_run(results_root, r, s) for r in prereg["replication_seeds"]]
    pf = prereg["primary_feature_model"]
    bundles = []
    for p in sorted((results_root / primary["run"]).glob("llm_bundle_*.json")):
        bundles.append(json.loads(p.read_text()))
    body = "".join(
        [
            f"<h1>abuse-triage-kit report</h1><p class=sub>Generated by <code>make demo</code> at commit "
            f"<code>{esc(commit)}</code>. Synthetic data only.</p>",
            f'<div class="banner"><b>Every LLM number in this report is {MOCK_TAG}.</b> It comes from '
            "MockLLM, which scores an account by hashing its prompt. It exists to test the plumbing "
            "and cost accounting, not to measure an LLM.</div>",
            _setup(prereg, primary, {}),
            _null(null),
            _signal(s1),
            _in_vs_held(primary, s1, prereg),
            _template({0.5: primary, 1.0: s1}),
            _budget(primary, prereg["review_budget_k"]),
            _cost(primary, bundles),
            _expected(prereg, primary, s1, null, pf),
            _replication(prereg, reps, primary, pf),
            _bundles(bundles, {}),
            CANT_TELL,
        ]
    )
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Abuse triage report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path
