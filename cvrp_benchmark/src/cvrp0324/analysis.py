from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ranksums


@dataclass(frozen=True)
class AnalysisBundle:
    raw_by_algorithm: dict[str, pd.DataFrame]
    raw_all: pd.DataFrame
    instance_summary_long: pd.DataFrame
    instance_summary_wide: pd.DataFrame
    benchmark_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    pairwise_gap_tests: pd.DataFrame
    tevc_detail: pd.DataFrame
    tevc_summary_by_family: pd.DataFrame
    tevc_summary_overall: pd.DataFrame


def analyze_experiment(
    raw_by_algorithm: dict[str, pd.DataFrame],
    *,
    algorithm_order: Iterable[str],
    baseline_algorithm: str,
    alpha: float,
) -> AnalysisBundle:
    algorithm_names = list(algorithm_order)
    normalized = {
        algorithm_name: normalize_raw_frame(frame)
        for algorithm_name, frame in raw_by_algorithm.items()
    }
    raw_all = pd.concat(
        [normalized[name] for name in algorithm_names if name in normalized],
        ignore_index=True,
    )
    raw_all = raw_all.sort_values(["Family", "Instance", "Algorithm", "Run"], kind="stable").reset_index(drop=True)

    instance_summary_long = build_instance_summary(raw_all)
    instance_summary_wide = build_instance_wide(instance_summary_long, algorithm_names)
    benchmark_summary = build_benchmark_summary(instance_summary_long)
    overall_summary = build_overall_summary(instance_summary_long)
    pairwise_gap_tests = build_pairwise_gap_tests(instance_summary_long, alpha)
    tevc_detail, tevc_summary_by_family, tevc_summary_overall = build_tevc_significance(
        raw_all,
        baseline_algorithm=baseline_algorithm,
        alpha=alpha,
    )

    return AnalysisBundle(
        raw_by_algorithm=normalized,
        raw_all=raw_all,
        instance_summary_long=instance_summary_long,
        instance_summary_wide=instance_summary_wide,
        benchmark_summary=benchmark_summary,
        overall_summary=overall_summary,
        pairwise_gap_tests=pairwise_gap_tests,
        tevc_detail=tevc_detail,
        tevc_summary_by_family=tevc_summary_by_family,
        tevc_summary_overall=tevc_summary_overall,
    )


def normalize_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        return normalized

    valid_series = normalized["Valid"]
    if valid_series.dtype != bool:
        normalized["Valid"] = (
            valid_series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)
        )

    numeric_columns = ["Run", "OptCost", "Cost", "GapPct", "TimeSec"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized["Algorithm"] = normalized["Algorithm"].astype(str)
    normalized["Family"] = normalized["Family"].astype(str)
    normalized["Instance"] = normalized["Instance"].astype(str)
    normalized["Message"] = normalized["Message"].fillna("").astype(str)
    return normalized


def build_instance_summary(raw_all: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["Algorithm", "Family", "Instance"]
    group_keys = raw_all[group_columns + ["OptCost"]].drop_duplicates(subset=group_columns)
    total_runs = raw_all.groupby(group_columns).size().rename("TotalRuns").reset_index()

    valid_rows = raw_all[raw_all["Valid"] & raw_all["Cost"].notna()].copy()
    valid_runs = valid_rows.groupby(group_columns).size().rename("ValidRuns").reset_index()

    if valid_rows.empty:
        summary = group_keys.merge(total_runs, on=group_columns, how="left")
        summary["ValidRuns"] = 0
        for column in [
            "MeanCost",
            "StdCost",
            "BestCost",
            "WorstCost",
            "MedianCost",
            "MeanGapPct",
            "StdGapPct",
            "BestGapPct",
            "MeanTimeSec",
            "StdTimeSec",
        ]:
            summary[column] = np.nan
        return summary.sort_values(["Family", "Instance", "Algorithm"], kind="stable").reset_index(drop=True)

    aggregates = valid_rows.groupby(group_columns).agg(
        MeanCost=("Cost", "mean"),
        StdCost=("Cost", _std0),
        BestCost=("Cost", "min"),
        WorstCost=("Cost", "max"),
        MedianCost=("Cost", "median"),
        MeanGapPct=("GapPct", "mean"),
        StdGapPct=("GapPct", _std0),
        BestGapPct=("GapPct", "min"),
        MeanTimeSec=("TimeSec", "mean"),
        StdTimeSec=("TimeSec", _std0),
    ).reset_index()

    summary = group_keys.merge(aggregates, on=group_columns, how="left")
    summary = summary.merge(valid_runs, on=group_columns, how="left")
    summary = summary.merge(total_runs, on=group_columns, how="left")
    summary["ValidRuns"] = summary["ValidRuns"].fillna(0).astype(int)
    summary["TotalRuns"] = summary["TotalRuns"].fillna(0).astype(int)
    summary["InvalidRuns"] = summary["TotalRuns"] - summary["ValidRuns"]

    return summary.sort_values(["Family", "Instance", "Algorithm"], kind="stable").reset_index(drop=True)


def build_instance_wide(instance_summary_long: pd.DataFrame, algorithm_order: list[str]) -> pd.DataFrame:
    key_frame = instance_summary_long[["Family", "Instance", "OptCost"]].drop_duplicates(subset=["Family", "Instance"])
    wide = key_frame.sort_values(["Family", "Instance"], kind="stable").reset_index(drop=True)

    selected_columns = [
        "MeanCost",
        "StdCost",
        "MeanGapPct",
        "StdGapPct",
        "BestCost",
        "ValidRuns",
    ]
    for algorithm_name in algorithm_order:
        subset = instance_summary_long[instance_summary_long["Algorithm"] == algorithm_name][
            ["Family", "Instance"] + selected_columns
        ].copy()
        subset = subset.rename(columns={column: f"{algorithm_name}_{column}" for column in selected_columns})
        wide = wide.merge(subset, on=["Family", "Instance"], how="left")

    return wide


def build_benchmark_summary(instance_summary_long: pd.DataFrame) -> pd.DataFrame:
    valid_instances = instance_summary_long[instance_summary_long["ValidRuns"] > 0].copy()
    if valid_instances.empty:
        return pd.DataFrame(
            columns=[
                "Algorithm",
                "Family",
                "SolvedInstances",
                "MeanGapPct",
                "StdGapPctAcrossInstances",
                "MeanBestGapPct",
                "MeanCostStd",
                "MeanTimeSec",
            ]
        )

    benchmark_summary = valid_instances.groupby(["Algorithm", "Family"]).agg(
        SolvedInstances=("Instance", "count"),
        MeanGapPct=("MeanGapPct", "mean"),
        StdGapPctAcrossInstances=("MeanGapPct", _std0),
        MeanBestGapPct=("BestGapPct", "mean"),
        MeanCostStd=("StdCost", "mean"),
        MeanTimeSec=("MeanTimeSec", "mean"),
    ).reset_index()
    return benchmark_summary.sort_values(["Family", "Algorithm"], kind="stable").reset_index(drop=True)


def build_overall_summary(instance_summary_long: pd.DataFrame) -> pd.DataFrame:
    valid_instances = instance_summary_long[instance_summary_long["ValidRuns"] > 0].copy()
    if valid_instances.empty:
        return pd.DataFrame(
            columns=[
                "Algorithm",
                "SolvedInstances",
                "MeanGapPct",
                "StdGapPctAcrossInstances",
                "MeanBestGapPct",
                "MeanCostStd",
                "MeanTimeSec",
            ]
        )

    overall_summary = valid_instances.groupby("Algorithm").agg(
        SolvedInstances=("Instance", "count"),
        MeanGapPct=("MeanGapPct", "mean"),
        StdGapPctAcrossInstances=("MeanGapPct", _std0),
        MeanBestGapPct=("BestGapPct", "mean"),
        MeanCostStd=("StdCost", "mean"),
        MeanTimeSec=("MeanTimeSec", "mean"),
    ).reset_index()
    return overall_summary.sort_values(["MeanGapPct", "Algorithm"], kind="stable").reset_index(drop=True)


def build_pairwise_gap_tests(instance_summary_long: pd.DataFrame, alpha: float) -> pd.DataFrame:
    valid_instances = instance_summary_long[instance_summary_long["ValidRuns"] > 0].copy()
    algorithms = list(valid_instances["Algorithm"].drop_duplicates())
    rows: list[dict[str, object]] = []

    for index, algorithm_1 in enumerate(algorithms):
        left = valid_instances[valid_instances["Algorithm"] == algorithm_1][["Family", "Instance", "MeanGapPct"]]
        for algorithm_2 in algorithms[index + 1 :]:
            right = valid_instances[valid_instances["Algorithm"] == algorithm_2][["Family", "Instance", "MeanGapPct"]]
            merged = left.merge(right, on=["Family", "Instance"], how="inner", suffixes=("_1", "_2")).dropna()
            if len(merged) < 3:
                continue
            statistic, p_value = ranksums(merged["MeanGapPct_1"], merged["MeanGapPct_2"])
            mean_gap_1 = float(merged["MeanGapPct_1"].mean())
            mean_gap_2 = float(merged["MeanGapPct_2"].mean())
            rows.append(
                {
                    "Algorithm_1": algorithm_1,
                    "Algorithm_2": algorithm_2,
                    "MeanGap_1": mean_gap_1,
                    "MeanGap_2": mean_gap_2,
                    "MeanDifference": mean_gap_1 - mean_gap_2,
                    "Statistic": float(statistic),
                    "PValue": float(p_value),
                    "Significant": bool(p_value < alpha),
                    "ComparedInstances": int(len(merged)),
                }
            )

    return pd.DataFrame(rows)


def build_tevc_significance(
    raw_all: pd.DataFrame,
    *,
    baseline_algorithm: str,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_rows = raw_all[raw_all["Valid"] & raw_all["Cost"].notna()].copy()
    baseline_rows = valid_rows[valid_rows["Algorithm"] == baseline_algorithm]
    competitors = [name for name in valid_rows["Algorithm"].drop_duplicates() if name != baseline_algorithm]

    detail_rows: list[dict[str, object]] = []
    for competitor in competitors:
        competitor_rows = valid_rows[valid_rows["Algorithm"] == competitor]
        candidate_pairs = baseline_rows[["Family", "Instance"]].drop_duplicates().merge(
            competitor_rows[["Family", "Instance"]].drop_duplicates(),
            on=["Family", "Instance"],
            how="inner",
        )
        for pair in candidate_pairs.itertuples(index=False):
            base_costs = baseline_rows[
                (baseline_rows["Family"] == pair.Family) & (baseline_rows["Instance"] == pair.Instance)
            ]["Cost"].dropna()
            competitor_costs = competitor_rows[
                (competitor_rows["Family"] == pair.Family) & (competitor_rows["Instance"] == pair.Instance)
            ]["Cost"].dropna()

            base_mean = float(base_costs.mean())
            competitor_mean = float(competitor_costs.mean())
            if len(base_costs) < 2 or len(competitor_costs) < 2:
                statistic = np.nan
                p_value = np.nan
                symbol = "="
                outcome = "insufficient_runs"
            else:
                statistic, p_value = ranksums(base_costs, competitor_costs)
                if p_value < alpha and base_mean < competitor_mean:
                    symbol = "+"
                    outcome = "baseline_better"
                elif p_value < alpha and base_mean > competitor_mean:
                    symbol = "-"
                    outcome = "baseline_worse"
                else:
                    symbol = "="
                    outcome = "no_significant_difference"

            detail_rows.append(
                {
                    "Baseline": baseline_algorithm,
                    "Competitor": competitor,
                    "Family": pair.Family,
                    "Instance": pair.Instance,
                    "BaselineMeanCost": base_mean,
                    "CompetitorMeanCost": competitor_mean,
                    "Statistic": float(statistic) if not pd.isna(statistic) else np.nan,
                    "PValue": float(p_value) if not pd.isna(p_value) else np.nan,
                    "Symbol": symbol,
                    "Outcome": outcome,
                }
            )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        empty_summary = pd.DataFrame(columns=["Competitor", "Family", "+", "=", "-", "ComparedInstances"])
        empty_overall = pd.DataFrame(columns=["Competitor", "+", "=", "-", "ComparedInstances"])
        return detail, empty_summary, empty_overall

    summary_by_family = _count_symbols(detail, ["Competitor", "Family"])
    summary_overall = _count_symbols(detail, ["Competitor"])
    return detail, summary_by_family, summary_overall


def _count_symbols(detail: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        detail.groupby(group_columns + ["Symbol"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for symbol in ("+", "=", "-"):
        if symbol not in summary.columns:
            summary[symbol] = 0
    compared = detail.groupby(group_columns).size().rename("ComparedInstances").reset_index()
    summary = compared.merge(summary, on=group_columns, how="left")
    ordered_columns = group_columns + ["+", "=", "-", "ComparedInstances"]
    return summary[ordered_columns].sort_values(group_columns, kind="stable").reset_index(drop=True)


def _std0(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=0))
