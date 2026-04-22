from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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
    instance_catalog: pd.DataFrame,
    instance_paths: list[Path],
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
    bundle.edge_type_summary.to_csv(summary_dir / "edge_type_summary.csv", index=False, float_format="%.6f")
    bundle.size_bucket_summary.to_csv(summary_dir / "size_bucket_summary.csv", index=False, float_format="%.6f")
    bundle.overall_summary.to_csv(summary_dir / "overall_summary.csv", index=False, float_format="%.6f")
    bundle.pairwise_gap_tests.to_csv(summary_dir / "pairwise_gap_tests.csv", index=False, float_format="%.6f")
    bundle.tevc_detail.to_csv(summary_dir / "tevc_detail.csv", index=False, float_format="%.6f")
    bundle.tevc_summary_overall.to_csv(summary_dir / "tevc_summary_overall.csv", index=False, float_format="%.6f")
    instance_catalog.to_csv(meta_dir / "instance_catalog.csv", index=False, float_format="%.6f")

    write_summary_workbook(summary_dir / "summary.xlsx", bundle, algorithm_order, instance_catalog)
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
        "selected_instance_count": len(instance_paths),
        "instances": [path.name for path in instance_paths],
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (meta_dir / "config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary_workbook(
    workbook_path: Path,
    bundle: AnalysisBundle,
    algorithm_order: list[str],
    instance_catalog: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        bundle.raw_all.to_excel(writer, sheet_name="raw_all", index=False)
        bundle.instance_summary_long.to_excel(writer, sheet_name="instance_long", index=False)
        bundle.instance_summary_wide.to_excel(writer, sheet_name="instance_wide", index=False)
        bundle.edge_type_summary.to_excel(writer, sheet_name="edge_type", index=False)
        bundle.size_bucket_summary.to_excel(writer, sheet_name="size_bucket", index=False)
        bundle.overall_summary.to_excel(writer, sheet_name="overall", index=False)
        bundle.pairwise_gap_tests.to_excel(writer, sheet_name="pairwise_gap", index=False)
        bundle.tevc_detail.to_excel(writer, sheet_name="tevc_detail", index=False)
        bundle.tevc_summary_overall.to_excel(writer, sheet_name="tevc_overall", index=False)
        instance_catalog.to_excel(writer, sheet_name="instance_catalog", index=False)

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
        "## Overall Summary",
        "",
        "```latex",
        _render_overall_summary_table(bundle.overall_summary),
        "```",
        "",
        "## Edge Type Summary",
        "",
        "```latex",
        _render_group_summary_table(bundle.edge_type_summary, "EdgeWeightType", algorithm_order, "Average gap by edge-weight type."),
        "```",
        "",
        "## Size Bucket Summary",
        "",
        "```latex",
        _render_group_summary_table(bundle.size_bucket_summary, "SizeBucket", algorithm_order, "Average gap by size bucket."),
        "```",
        "",
        "## Per-Instance Appendix",
        "",
        "```latex",
        _render_instance_table(bundle.instance_summary_long, algorithm_order),
        "```",
    ]
    return "\n".join(sections).strip() + "\n"


def render_tevc_markdown(bundle: AnalysisBundle, algorithm_order: list[str], baseline_algorithm: str) -> str:
    competitors = [name for name in algorithm_order if name != baseline_algorithm]
    symbol_matrix = (
        _render_tevc_symbol_matrix(bundle.tevc_detail, competitors)
        if not bundle.tevc_detail.empty
        else r"\textit{No significance details were generated.}"
    )
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
        _render_tevc_summary_table(bundle.tevc_summary_overall),
        "```",
        "",
        "## Symbol Matrix",
        "",
        "```latex",
        symbol_matrix,
        "```",
    ]
    return "\n".join(sections).strip() + "\n"


def _render_overall_summary_table(overall_summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Overall statistics aggregated over TSP instances.}",
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


def _render_group_summary_table(
    summary: pd.DataFrame,
    group_column: str,
    algorithm_order: list[str],
    caption: str,
) -> str:
    rows = []
    for group_value in summary[group_column].dropna().unique():
        row_frame = summary[summary[group_column] == group_value]
        rows.append((group_value, _format_metric_cells(row_frame, algorithm_order, "MeanGapPct", "StdGapPctAcrossInstances")))

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{l{'c' * len(algorithm_order)}}}",
        r"\toprule",
        _escape_latex(group_column) + " & " + " & ".join(_escape_latex(name) for name in algorithm_order) + r" \\",
        r"\midrule",
    ]
    for group_value, cells in rows:
        lines.append(_escape_latex(str(group_value)) + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _render_instance_table(instance_summary_long: pd.DataFrame, algorithm_order: list[str]) -> str:
    lines = [
        r"\begin{longtable}{" + "lcc" + "c" * len(algorithm_order) + "}",
        r"\caption{Per-instance mean cost $\pm$ std.} \\",
        r"\toprule",
        "Instance & Dim. & Type & " + " & ".join(_escape_latex(name) for name in algorithm_order) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        "Instance & Dim. & Type & " + " & ".join(_escape_latex(name) for name in algorithm_order) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for instance_name in instance_summary_long["Instance"].drop_duplicates():
        row_frame = instance_summary_long[instance_summary_long["Instance"] == instance_name]
        dimension = int(row_frame["Dimension"].iloc[0])
        edge_type = row_frame["EdgeWeightType"].iloc[0]
        cells = _format_metric_cells(row_frame, algorithm_order, "MeanCost", "StdCost")
        lines.append(
            _escape_latex(instance_name)
            + " & "
            + str(dimension)
            + " & "
            + _escape_latex(edge_type)
            + " & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def _render_tevc_summary_table(summary: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TEVC-style Wilcoxon rank-sum comparison against the baseline.}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Competitor & + & = & - & N \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                [
                    _escape_latex(str(row["Competitor"])),
                    str(int(row["+"])),
                    str(int(row["="])),
                    str(int(row["-"])),
                    str(int(row["ComparedInstances"])),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _render_tevc_symbol_matrix(detail: pd.DataFrame, competitors: list[str]) -> str:
    pivot = detail.pivot(index="Instance", columns="Competitor", values="Symbol")
    pivot = pivot.reindex(columns=competitors)
    lines = [
        r"\begin{longtable}{" + "l" + "c" * len(competitors) + "}",
        r"\caption{Significance symbols by instance.} \\",
        r"\toprule",
        "Instance & " + " & ".join(_escape_latex(name) for name in competitors) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        "Instance & " + " & ".join(_escape_latex(name) for name in competitors) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for instance_name, row in pivot.iterrows():
        cells = [row.get(competitor, "--") if pd.notna(row.get(competitor, "")) else "--" for competitor in competitors]
        lines.append(_escape_latex(str(instance_name)) + " & " + " & ".join(cells) + r" \\")
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
        algorithm_name: lookup.get(algorithm_name, {}).get(mean_column, pd.NA)
        for algorithm_name in algorithm_order
    }
    valid_values = [float(value) for value in means.values() if pd.notna(value)]
    best_mean = min(valid_values) if valid_values else None

    cells = []
    for algorithm_name in algorithm_order:
        row = lookup.get(algorithm_name)
        if row is None or pd.isna(row.get(mean_column, pd.NA)):
            cells.append("--")
            continue
        cell = f"{_format_number(row[mean_column])} $\\pm$ {_format_number(row.get(std_column, pd.NA))}"
        if best_mean is not None and abs(float(row[mean_column]) - best_mean) <= 1e-9:
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    return cells


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
