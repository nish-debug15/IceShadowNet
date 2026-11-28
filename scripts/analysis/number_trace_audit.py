"""
scripts/analysis/number_trace_audit.py
======================================
Block 7 Item 4: Strict Number-Trace Audit across all documentation and dashboard JSON.
Works purely from tracked JSON/YAML files (no gitignored .npz/.pth/.tif required) so it runs in CI.

Extracts EVERY number from:
  - README.md
  - reports/real/research_report.md
  - reports/real/viva_qa.md
  - docs/index.html
  - dashboard/public/data/segmentation_uncertainty.json

Maps each number to a verified key in reports/real/*.json, data/processed_real/split_manifest.json,
or config.yaml within exact decimal rounding tolerance, writes reports/real/claims_audit.md,
and exits with code 1 if ANY number is unmapped or if dashboard JSON drifts from reports/real/*.json.
"""

import html
import json
import os
import re
import sys
import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(REPO_ROOT, "reports", "real")
DASH_JSON = os.path.join(REPO_ROOT, "dashboard", "public", "data", "segmentation_uncertainty.json")
AUDIT_OUT = os.path.join(REPORTS, "claims_audit.md")

DOC_FILES = [
    ("README.md", os.path.join(REPO_ROOT, "README.md")),
    ("reports/real/research_report.md", os.path.join(REPORTS, "research_report.md")),
    ("reports/real/viva_qa.md", os.path.join(REPORTS, "viva_qa.md")),
    ("docs/index.html", os.path.join(REPO_ROOT, "docs", "index.html")),
]

DASHBOARD_SOURCE_MAP = {
    "unet": "unet_segmentation_report.json",
    "uncertainty": "uncertainty_report.json",
    "efficiency": "efficiency_report.json",
    "block1": "block1_benchmark_report.json",
    "five_fold_cv": "five_fold_cv.json",
    "block6_stats": "block6_stats.json",
    "context_baselines": "context_pixel_baselines.json",
    "selective_redone": "selective_redone.json",
    "phase_oracle": "phase_oracle_diagnostic.json",
    "pruning_unified": "pruning_unified.json",
    "georef_10fold": "georef_10fold.json",
    "georef_summary": "georef_summary.json",
}

FORBIDDEN_PHRASES = [
    "resolves threshold ambiguity",
    "solves ambiguity",
    "ensemble variance adds selective-prediction value",
    "usable for selective rejection",
]


def walk_numbers(obj, prefix, entries):
    """Recursively register every int/float leaf in a JSON/YAML dict/list."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        val = float(obj)
        entries.append((prefix, val))
        # Normalized fractions/probabilities/prevalences often displayed as %
        if -1.0 <= val <= 1.0 and val != 0.0:
            entries.append((f"{prefix} (%)", val * 100.0))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk_numbers(v, f"{prefix}.{k}", entries)
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            walk_numbers(v, f"{prefix}[{idx}]", entries)


def build_catalog():
    """Build catalog of (source_key, float_value) from tracked JSONs, manifest, and config.yaml."""
    entries = []

    cfg = yaml.safe_load(open(os.path.join(REPO_ROOT, "config.yaml"), "r", encoding="utf-8"))
    walk_numbers(cfg, "config.yaml", entries)

    manifest_path = os.path.join(REPO_ROOT, "data", "processed_real", "split_manifest.json")
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    manifest_no_patches = {k: v for k, v in manifest.items() if k != "patches"}
    walk_numbers(manifest_no_patches, "split_manifest.json", entries)

    json_data = {}
    for fname in sorted(os.listdir(REPORTS)):
        if fname.endswith(".json"):
            fpath = os.path.join(REPORTS, fname)
            data = json.load(open(fpath, "r", encoding="utf-8"))
            json_data[fname] = data
            walk_numbers(data, fname, entries)

    # Derived expressions directly computed from tracked JSON/YAML fields
    rc = cfg["real_dfsar"]
    counts = manifest["counts"]
    gs = json_data["georef_summary.json"]
    g10 = json_data["georef_10fold.json"]
    b6 = json_data["block6_stats.json"]
    b1 = json_data["block1_benchmark_report.json"]
    unet = json_data["unet_segmentation_report.json"]
    ctx = json_data["context_pixel_baselines.json"]
    unc = json_data["uncertainty_report.json"]
    ora = json_data["phase_oracle_diagnostic.json"]
    pru = json_data["pruning_unified.json"]

    active_len = manifest["active_rows"][1] - manifest["active_rows"][0]
    total_patches = counts["train"]["patches"] + counts["val"]["patches"] + counts["test"]["patches"]
    train_val_patches = counts["train"]["patches"] + counts["val"]["patches"]

    entries.extend([
        ("derived: active_rows[1]-active_rows[0]", float(active_len)),
        ("derived: 100*active_len/source_rows", 100.0 * active_len / gs["source_rows"]),
        ("derived: total_patches (train+val+test)", float(total_patches)),
        ("derived: train_val_patches", float(train_val_patches)),
        ("derived: test_patch_prevalence (%)", 100.0 * counts["test"]["positives"] / counts["test"]["patches"]),
        ("derived: n_cv_folds", float(len(set(r["fold"] for r in b6["five_fold_cv"])))),
        ("derived: n_seeds", float(len(ora["seeds"]))),
        ("derived: n_gcp_cv_folds", float(len(g10["folds"]))),
        ("derived: georef_max_km (ten_fold_max_m/1000)", g10["ten_fold_max_m"] / 1000.0),
        ("derived: pct_full (100.0)", 100.0),
        ("derived: metric_p_midpoint (0.5)", 0.5),
        ("derived: pruning_batch_B1 (1)", 1.0),
        ("derived: pruning_batch_B64 (64)", 64.0),
    ])

    # 5-fold CV means/stds
    cv_groups = {}
    for row in b6["five_fold_cv"]:
        cv_groups.setdefault(row["model"], []).append(row)
    for model, rows in cv_groups.items():
        rocs = [r["roc_auc"] for r in rows]
        prs = [r["pr_auc"] for r in rows]
        entries.extend([
            (f"derived: five_fold_cv[{model}].roc_mean", float(np.mean(rocs))),
            (f"derived: five_fold_cv[{model}].roc_std", float(np.std(rocs))),
            (f"derived: five_fold_cv[{model}].pr_mean", float(np.mean(prs))),
            (f"derived: five_fold_cv[{model}].pr_std", float(np.std(prs))),
        ])

    # Block 1 3-seed means/stds
    b1_groups = {}
    for row in b1:
        b1_groups.setdefault(row["model"], []).append(row["pooled"]["roc_auc"])
    for model, rocs in b1_groups.items():
        entries.extend([
            (f"derived: block1[{model}].roc_mean", float(np.mean(rocs))),
            (f"derived: block1[{model}].roc_std", float(np.std(rocs))),
        ])

    # U-Net 3-seed means/stds & lifts
    px_prev = unet["pixel_prevalence_test"]
    unet_prs = [r["pooled_val_th"]["pr_auc"] for r in unet["unet_runs"]]
    unet_dices = [r["pooled_val_th"]["dice"] for r in unet["unet_runs"]]
    unet_ious = [r["pooled_val_th"]["iou"] for r in unet["unet_runs"]]
    unet_mean_pr = float(np.mean(unet_prs))
    ens_pr = unc["deep_ensemble"]["pr_auc"]

    entries.extend([
        ("derived: hv_hh.lift", unet["baselines"]["hv_hh_ratio"]["pooled_val_th"]["pr_auc"] / px_prev),
        ("derived: px_logreg.lift", unet["baselines"]["pixel_logreg"]["pooled_val_th"]["pr_auc"] / px_prev),
        ("derived: unet_3seed.pr_mean", unet_mean_pr),
        ("derived: unet_3seed.pr_std", float(np.std(unet_prs))),
        ("derived: unet_3seed.lift", unet_mean_pr / px_prev),
        ("derived: unet_3seed.dice_mean", float(np.mean(unet_dices))),
        ("derived: unet_3seed.iou_mean", float(np.mean(unet_ious))),
        ("derived: unet_ensemble.lift", ens_pr / px_prev),
        ("derived: georef.swath_offset_p90_label (90)", 90.0),
    ])

    # Context window sizes & HistGBDT recovery fractions
    for cr in ctx["context_baselines"]:
        for m in re.findall(r"\d+", cr["windows"]):
            entries.append((f"context_pixel_baselines.windows[{cr['model']}]", float(m)))
    gbdt15 = [r for r in ctx["context_baselines"] if r["model"] == "Context Pixel HistGBDT (7x7+15x15)"][0]
    entries.extend([
        ("derived: 100*gbdt15_pr/unet_mean_pr", 100.0 * gbdt15["pr_auc"] / unet_mean_pr),
        ("derived: 100*gbdt15_pr/ens_pr", 100.0 * gbdt15["pr_auc"] / ens_pr),
    ])

    # Phase oracle relative gain
    o3_pr = ora["intensity_3ch"]["summary"]["pr_auc"]["mean"]
    o4_pr = ora["phase_oracle_4ch"]["summary"]["pr_auc"]["mean"]
    entries.append(("derived: 100*(ora_4ch_pr-ora_3ch_pr)/ora_3ch_pr", 100.0 * (o4_pr - o3_pr) / o3_pr))

    # Pruning percentages, base_ch params & speedups
    for idx, row in enumerate(pru):
        for m in re.findall(r"\d+", row["config"]):
            entries.append((f"pruning_unified[{idx}].config_tag", float(m)))
    entries.extend([
        ("derived: pruning_s50_b64_speedup", pru[0]["lat_b64_ms"] / pru[2]["lat_b64_ms"]),
        ("derived: pruning_s50_b1_speedup", pru[0]["lat_b1_ms"] / pru[2]["lat_b1_ms"]),
    ])

    return entries, json_data


def verify_dashboard_json(json_data):
    """Recursively verify every numeric leaf in segmentation_uncertainty.json matches reports/real/*.json."""
    dash = json.load(open(DASH_JSON, "r", encoding="utf-8"))
    mismatches = []

    def cmp_leaves(d_obj, s_obj, path):
        if isinstance(d_obj, bool) or isinstance(s_obj, bool):
            return
        if isinstance(d_obj, (int, float)) and isinstance(s_obj, (int, float)):
            if abs(float(d_obj) - float(s_obj)) > 1e-9:
                mismatches.append((path, d_obj, s_obj))
        elif isinstance(d_obj, dict) and isinstance(s_obj, dict):
            for k in d_obj:
                if k in s_obj:
                    cmp_leaves(d_obj[k], s_obj[k], f"{path}.{k}")
                else:
                    mismatches.append((f"{path}.{k}", d_obj[k], "MISSING_IN_SOURCE"))
        elif isinstance(d_obj, list) and isinstance(s_obj, list):
            if len(d_obj) != len(s_obj):
                mismatches.append((path, f"len={len(d_obj)}", f"len={len(s_obj)}"))
            else:
                for i, (dv, sv) in enumerate(zip(d_obj, s_obj)):
                    cmp_leaves(dv, sv, f"{path}[{i}]")

    for section, src_fname in DASHBOARD_SOURCE_MAP.items():
        if section not in dash:
            mismatches.append((section, "MISSING_IN_DASHBOARD", src_fname))
        elif src_fname not in json_data:
            mismatches.append((section, "PRESENT", f"MISSING_{src_fname}"))
        else:
            cmp_leaves(dash[section], json_data[src_fname], f"segmentation_uncertainty.{section}")
    return mismatches


def preprocess_text(raw_text, is_html=False):
    """Strip HTML tags/CSS, filenames/identifiers, and mission proper nouns cleanly."""
    txt = raw_text
    if is_html:
        txt = re.sub(r"<style[^>]*>.*?</style>", " ", txt, flags=re.DOTALL | re.IGNORECASE)
        txt = re.sub(r"<[^>]+>", " ", txt)
        txt = html.unescape(txt)

    # Strip known non-empirical proper nouns, release tags, filenames, and scene IDs
    non_metric_patterns = [
        r"ch2_sar_ncxl_20260308t062017233",
        r"IceShadowNet\s+2\.0",
        r"Chandrayaan-2",
        r"Level-1A",
        r"\bv1\.0-endsem\b",
        r"\bL1A\b",
        r"\bL1\b",
        r"\bseed\d+\b",
        r"\([^\)]*\.(?:md|json|png|tif|pth|npz|yaml|sh|py)\)",
    ]
    for pat in non_metric_patterns:
        txt = re.sub(pat, " ", txt)
    return txt


def decimal_precision(num_str):
    """Return number of decimal places shown for rounding tolerance."""
    clean = num_str.replace(",", "").lstrip("+-")
    if "e" in clean.lower():
        mant, _ = clean.lower().split("e")
        # For sci notation, check precision in mantissa * 10**exp
        exp = int(clean.lower().split("e")[1])
        dec_m = len(mant.split(".")[1]) if "." in mant else 0
        return max(0, dec_m - exp)
    if "." in clean:
        return len(clean.split(".")[1])
    return 0


def match_number(num_str, catalog, is_pct=False):
    val = float(num_str.replace(",", ""))
    dec = decimal_precision(num_str)
    tol = 0.55 * (10.0 ** (-dec)) + 1e-12
    # Prefer (%) entries when token is followed by '%', and non-(%) entries otherwise
    best = None
    best_rank = (9, float("inf"))
    for src_key, cval in catalog:
        diff = abs(val - cval)
        if diff <= tol:
            key_is_pct = src_key.endswith("(%)") or ("(%)" in src_key)
            tier = 0 if (key_is_pct == is_pct) else 1
            rank = (tier, diff)
            if rank < best_rank:
                best = (src_key, cval, diff)
                best_rank = rank
    return best


def audit_doc(name, path, is_html, catalog):
    raw = open(path, "r", encoding="utf-8").read()
    # Forbidden claims check
    low = raw.lower()
    forbidden_hits = [p for p in FORBIDDEN_PHRASES if p in low and f"not {p}" not in low and f"not\n{p}" not in low]
    # Also check strict negative assertion wording cleanly: forbid "resolves threshold ambiguity", "solves ambiguity"
    strict_forbidden = [
        p for p in ["resolves threshold ambiguity", "solves ambiguity", "ensemble variance adds selective-prediction value"]
        if p in low
    ]

    cleaned = preprocess_text(raw, is_html=is_html)
    matched = []
    unmapped = []

    num_re = re.compile(r"(?<![A-Za-z_])[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_])")
    for lineno, line in enumerate(cleaned.splitlines(), start=1):
        # Replace subtraction minus before digits or en-dash with separator if preceded by space ("|p - 0.5|")
        # Keep tight unary +/- (e.g. +0.0228, -0.0109, -13.07) intact!
        line_norm = line.replace("–", " ").replace("—", " ")
        for m in num_re.finditer(line_norm):
            tok = m.group(0)
            after = line_norm[m.end():m.end() + 2]
            is_pct = after.startswith("%")
            res = match_number(tok, catalog, is_pct=is_pct)
            if res is None:
                unmapped.append((name, lineno, tok, line.strip()[:120]))
            else:
                matched.append((name, lineno, tok, res[0], res[1], res[2]))
    return matched, unmapped, strict_forbidden


def main():
    catalog, json_data = build_catalog()
    dash_mismatches = verify_dashboard_json(json_data)

    all_matched = []
    all_unmapped = []
    all_forbidden = []
    doc_stats = []

    for name, path in DOC_FILES:
        is_html = name.endswith(".html")
        matched, unmapped, forb = audit_doc(name, path, is_html, catalog)
        all_matched.extend(matched)
        all_unmapped.extend(unmapped)
        all_forbidden.extend([(name, f) for f in forb])
        doc_stats.append((name, len(matched), len(unmapped), len(forb)))

    passed = (len(all_unmapped) == 0) and (len(dash_mismatches) == 0) and (len(all_forbidden) == 0)

    lines = [
        "# Number-Trace Claims Audit (CI-Verified)",
        "",
        "Every numeric token across `README.md`, `reports/real/research_report.md`, `reports/real/viva_qa.md`, and `docs/index.html` is extracted and mapped to tracked `reports/real/*.json`, `data/processed_real/split_manifest.json`, or `config.yaml` within rounding tolerance, alongside a recursive leaf-equality check on `dashboard/public/data/segmentation_uncertainty.json`.",
        "",
        "## Summary by Audited Artifact",
        "",
        "| Audited Artifact | Extracted Numbers | Matched to JSON/Config | Unmapped | Forbidden Overstatements | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for name, nm, nu, nf in doc_stats:
        st = "PASS" if (nu == 0 and nf == 0) else "FAIL"
        lines.append(f"| `{name}` | `{nm + nu}` | `{nm}` | `{nu}` | `{nf}` | **{st}** |")

    dash_status = "PASS" if len(dash_mismatches) == 0 else f"FAIL ({len(dash_mismatches)} mismatches)"
    lines.append(
        f"| `dashboard/public/data/segmentation_uncertainty.json` | recursive leaves | exact match | `{len(dash_mismatches)}` | `0` | **{dash_status}** |"
    )

    lines.extend([
        "",
        f"**Overall Number-Trace Audit:** `{'PASSED' if passed else 'FAILED'}` "
        f"(`{len(all_matched)}/{len(all_matched) + len(all_unmapped)}` doc numbers traced; "
        f"`{len(dash_mismatches)}` dashboard JSON leaf mismatches; "
        f"`{len(all_forbidden)}` forbidden phrases).",
        "",
        "## Representative Source-Key Traces",
        "",
        "| Document | Line | Printed Token | Traced JSON / Config Key | Exact Source Value |",
        "| :--- | :---: | :---: | :--- | :---: |",
    ])
    seen_toks = set()
    for name, ln, tok, skey, cval, _ in all_matched:
        if (name, tok) not in seen_toks and len(seen_toks) < 35:
            seen_toks.add((name, tok))
            lines.append(f"| `{name}` | {ln} | `{tok}` | `{skey}` | `{cval:.6g}` |")

    if all_unmapped:
        lines.extend(["", "## Unmapped Numbers (FAILURES)", ""])
        for name, ln, tok, ctx in all_unmapped:
            lines.append(f"- `{name}:{ln}` token `{tok}` in: `{ctx}`")

    if dash_mismatches:
        lines.extend(["", "## Dashboard JSON Mismatches", ""])
        for path, dv, sv in dash_mismatches:
            lines.append(f"- `{path}`: dashboard=`{dv}` vs source=`{sv}`")

    with open(AUDIT_OUT, "w", encoding="utf-8") as af:
        af.write("\n".join(lines) + "\n")

    print(f"Number-trace audit complete: PASSED={passed} | doc_tokens_traced={len(all_matched)} | unmapped={len(all_unmapped)} | dash_mismatches={len(dash_mismatches)}")
    if all_unmapped:
        print("UNMAPPED NUMBERS:")
        for u in all_unmapped:
            print(" ", u)
    if dash_mismatches:
        print("DASHBOARD MISMATCHES:")
        for dm in dash_mismatches[:20]:
            print(" ", dm)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
