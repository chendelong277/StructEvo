from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import AnalysisBundle


def create_experiment_dir(output_root: Path, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(label) or "experiment"
    experiment_dir = output_root / f"{timestamp}_{slug}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def write_outputs(
    *,
    output_dir: Path,
    bundle: AnalysisBundle,
    config_payload: dict[str, Any],
    algorithm_order: list[str],
    baseline_algorithm: str,
    source: dict[str, Any],
    instance_paths: list[Path] | None = None,
) -> None:
    summary_dir = output_dir / "summary"
    tables_dir = output_dir / "tables"
    meta_dir = output_dir / "meta"
    raw_dir = output_dir / "raw_runs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    bundle.raw_all.to_csv(summary_dir / "raw_all.csv", index=False, float_format="%.6f")
    bundle.instance_summary_long.to_csv(summary_dir / "instance_summary_long.csv", index=False, float_format="%.6f")
    bundle.instance_summary_wide.to_csv(summary_dir / "instance_summary_wide.csv", index=False, float_format="%.6f")
    bundle.benchmark_summary.to_csv(summary_dir / "benchmark_summary.csv", index=False, float_format="%.6f")
    bundle.overall_summary.to_csv(summary_dir / "overall_summary.csv", index=False, float_format="%.6f")
    bundle.pairwise_gap_tests.to_csv(summary_dir / "pairwise_gap_tests.csv", index=False, float_format="%.6f")
    bundle.tevc_detail.to_csv(summary_dir / "tevc_detail.csv", index=False, float_format="%.6f")
    bundle.tevc_summary_by_family.to_csv(summary_dir / "tevc_summary_by_family.csv", index=False, float_format="%.6f")
    bundle.tevc_summary_overall.to_csv(summary_dir / "tevc_summary_overall.csv", index=False, float_format="%.6f")

    write_summary_workbook(summary_dir / "summary.xlsx", bundle, algorithm_order)
    (tables_dir / "latex_tables.md").write_text(
        render_latex_tables(bundle, algorithm_order),
        encoding="utf-8",
    )
    (tables_dir / "tevc_significance.md").write_text(
        render_tevc_markdown(bundle, algorithm_order, baseline_algorithm),
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_algorithm": baseline_algorithm,
        "algorithms": algorithm_order,
        "source": source,
        "config": config_payload,
        "instance_count": len(instance_paths or []),
        "instances": [
            f"{path.parent.name}/{path.name}"
            for path in (instance_paths or [])
        ],
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (meta_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_workbook(workbook_path: Path, bundle: AnalysisBundle, algorithm_order: list[str]) -> None:
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        bundle.raw_all.to_excel(writer, sheet_name="raw_all", index=False)
        bundle.instance_summary_long.to_excel(writer, sheet_name="instance_long", index=False)
        bundle.instance_summary_wide.to_excel(writer, sheet_name="instance_wide", index=False)
        bundle.benchmark_summary.to_excel(writer, sheet_name="benchmark", index=False)
        bundle.overall_summary.to_excel(writer, sheet_name="overall", index=False)
        bundle.pairwise_gap_tests.to_excel(writer, sheet_name="pairwise_gap", index=False)
        bundle.tevc_detail.to_excel(writer, sheet_name="tevc_detail", index=False)
        bundle.tevc_summary_by_family.to_excel(writer, sheet_name="tevc_family", index=False)
        bundle.tevc_summary_overall.to_excel(writer, sheet_name="tevc_overall", index=False)
        for algorithm_name in algorithm_order:
            frame = bundle.raw_by_algorithm.get(algorithm_name)
            if frame is None:
                continue
            frame.to_excel(writer, sheet_name=_sheet_name(f"raw_{algorithm_name}"), index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            for column in worksheet.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column[:200])
                worksheet.column_dimensions[column[0].column_letter].width = min(max(max_length + 2, 12), 40)


def render_latex_tables(bundle: AnalysisBundle, algorithm_order: list[str]) -> str:
    sections = [
        "# Latex Tables",
        "",
        "## Benchmark Summary",
        "",
        "```latex",
        _render_benchmark_gap_table(bundle.benchmark_summary, bundle.overall_summary, algorithm_order),
        "```",
        "",
        "## Overall Summary",
        "",
        "```latex",
        _render_overall_summary_table(bundle.overall_summary),
        "```",
    ]

    for family in sorted(bundle.instance_summary_long["Family"].dropna().unique()):
        sections.extend(
            [
                "",
                f"## Appendix: {family} Instances",
                "",
                "```latex",
                _render_family_instance_table(bundle.instance_summary_long, family, algorithm_order),
                "```",
            ]
        )

    return "\n".join(sections).strip() + "\n"


def render_tevc_markdown(bundle: AnalysisBundle, algorithm_order: list[str], baseline_algorithm: str) -> str:
    competitors = [name for name in algorithm_order if name != baseline_algorithm]
    sections = [
        "# TEVC-Style Significance",
        "",
        f"Baseline algorithm: `{baseline_algorithm}`",
        "",
        "Symbol convention: `+` means the baseline is significantly better, `-` means the baseline is significantly worse, `=` means no significant difference.",
        "",
        "## Overall Win/Tie/Loss",
        "",
        "```latex",
        _render_tevc_summary_table(bundle.tevc_summary_overall, include_family=False),
        "```",
        "",
        "## Win/Tie/Loss By Family",
        "",
        "```latex",
        _render_tevc_summary_table(bundle.tevc_summary_by_family, include_family=True),
        "```",
    ]

    for family in sorted(bundle.tevc_detail["Family"].dropna().unique()) if not bundle.tevc_detail.empty else []:
        sections.extend(
            [
                "",
                f"## Symbol Matrix: {family}",
                "",
                "```latex",
                _render_tevc_symbol_matrix(bundle.tevc_detail, family, competitors),
                "```",
            ]
        )

    return "\n".join(sections).strip() + "\n"


def _render_benchmark_gap_table(
    benchmark_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    algorithm_order: list[str],
) -> str:
    rows = []
    for family in sorted(benchmark_summary["Family"].dropna().unique()):
        row_frame = benchmark_summary[benchmark_summary["Family"] == family]
        rows.append((family, _format_metric_cells(row_frame, algorithm_order, "MeanGapPct", "StdGapPctAcrossInstances")))

    if not overall_summary.empty:
        rows.append(
            (
                "Overall",
                _format_metric_cells(overall_summary, algorithm_order, "MeanGapPct", "StdGapPctAcrossInstances", key="Algorithm"),
            )
        )

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Average gap (\%) across benchmark families.}",
        rf"\begin{{tabular}}{{l{'c' * len(algorithm_order)}}}",
        r"\toprule",
        "Family & " + " & ".join(_escape_latex(name) for name in algorithm_order) + r" \\",
        r"\midrule",
    ]
    for family, cells in rows:
        lines.append(_escape_latex(family) + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _render_overall_summary_table(overall_summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Overall statistics aggregated over instances.}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Algorithm & Instances & Mean Gap (\%) & Std Gap (\%) & Mean Best Gap (\%) & Mean Time (s) \\",
        r"\midrule",
    ]
    for row in overall_summary.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    _escape_latex(str(row.Algorithm)),
                    str(int(row.SolvedInstances)),
                    _format_number(row.MeanGapPct),
                    _format_number(row.StdGapPctAcrossInstances),
                    _format_number(row.MeanBestGapPct),
                    _format_number(row.MeanTimeSec),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _render_family_instance_table(
    instance_summary_long: pd.DataFrame,
    family: str,
    algorithm_order: list[str],
) -> str:
    family_frame = instance_summary_long[instance_summary_long["Family"] == family].copy()
    family_frame = family_frame.sort_values(["Instance", "Algorithm"], kind="stable")
    instances = list(family_frame["Instance"].drop_duplicates())

    lines = [
        r"\begin{longtable}{" + "l" + "c" * (len(algorithm_order) + 1) + "}",
        rf"\caption{{Per-instance mean cost $\pm$ std for family {_escape_latex(family)}.}} \\",
        r"\toprule",
        "Instance & Opt. & " + " & ".join(_escape_latex(name) for name in algorithm_order) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        "Instance & Opt. & " + " & ".join(_escape_latex(name) for name in algorithm_order) + r" \\",
        r"\midrule",
        r"\endhead",
    ]

    for instance in instances:
        row_frame = family_frame[family_frame["Instance"] == instance]
        opt_cost = row_frame["OptCost"].dropna().iloc[0] if row_frame["OptCost"].notna().any() else np.nan
        cells = _format_metric_cells(row_frame, algorithm_order, "MeanCost", "StdCost", key="Algorithm")
        lines.append(
            _escape_latex(instance)
            + " & "
            + _format_number(opt_cost)
            + " & "
            + " & ".join(cells)
            + r" \\"
        )

    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _render_tevc_summary_table(summary: pd.DataFrame, include_family: bool) -> str:
    headers = ["Competitor"]
    column_spec = "l"
    if include_family:
        headers.append("Family")
        column_spec += "l"
    headers.extend(["+", "=", "-", "N"])
    column_spec += "rrrr"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TEVC-style Wilcoxon rank-sum comparison against the baseline.}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        cells = [_escape_latex(str(row["Competitor"]))]
        if include_family:
            cells.append(_escape_latex(str(row["Family"])))
        cells.extend(
            [
                str(int(row["+"])),
                str(int(row["="])),
                str(int(row["-"])),
                str(int(row["ComparedInstances"])),
            ]
        )
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _render_tevc_symbol_matrix(detail: pd.DataFrame, family: str, competitors: list[str]) -> str:
    family_frame = detail[detail["Family"] == family].copy()
    pivot = family_frame.pivot(index="Instance", columns="Competitor", values="Symbol")
    pivot = pivot.reindex(columns=competitors)
    lines = [
        r"\begin{longtable}{" + "l" + "c" * len(competitors) + "}",
        rf"\caption{{Significance symbols for family {_escape_latex(family)}.}} \\",
        r"\toprule",
        "Instance & " + " & ".join(_escape_latex(name) for name in competitors) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        "Instance & " + " & ".join(_escape_latex(name) for name in competitors) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for instance, row in pivot.iterrows():
        cells = [row.get(competitor, "--") if pd.notna(row.get(competitor, np.nan)) else "--" for competitor in competitors]
        lines.append(_escape_latex(str(instance)) + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _format_metric_cells(
    frame: pd.DataFrame,
    algorithm_order: list[str],
    mean_column: str,
    std_column: str,
    *,
    key: str = "Algorithm",
) -> list[str]:
    lookup = frame.set_index(key).to_dict(orient="index")
    means = {
        algorithm_name: lookup.get(algorithm_name, {}).get(mean_column, np.nan)
        for algorithm_name in algorithm_order
    }
    best_mean = min((value for value in means.values() if pd.notna(value)), default=np.nan)
    cells = []
    for algorithm_name in algorithm_order:
        row = lookup.get(algorithm_name)
        if row is None or pd.isna(row.get(mean_column, np.nan)):
            cells.append("--")
            continue
        cell = _format_mean_std(row[mean_column], row.get(std_column, np.nan))
        if pd.notna(best_mean) and abs(row[mean_column] - best_mean) <= 1e-9:
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    return cells


def _format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{_format_number(mean_value)} $\\pm$ {_format_number(std_value)}"


def _format_number(value: float, decimals: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = text
    for original, replacement in replacements.items():
        escaped = escaped.replace(original, replacement)
    return escaped


def _sheet_name(value: str) -> str:
    return re.sub(r"[\[\]:*?/\\]", "_", value)[:31]
