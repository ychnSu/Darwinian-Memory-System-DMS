#!/usr/bin/env python3
"""Generate DMS evolution metrics and SVG charts from a results.csv file."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re
from typing import Iterable


def _float(value: object) -> float | None:
  if value in ("", None):
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _load_rows(path: Path) -> list[dict[str, object]]:
  with path.open(newline="", encoding="utf-8") as handle:
    rows = [
        row for row in csv.DictReader(handle)
        if _float(row.get("SA")) is not None
    ]
  cumulative_success = 0.0
  per_task_success: dict[str, float] = {}
  per_task_count: dict[str, int] = {}
  enriched: list[dict[str, object]] = []
  for index, row in enumerate(rows, start=1):
    task = str(row.get("task_template") or "unknown")
    success = float(_float(row.get("SA")) or 0.0)
    cumulative_success += success
    per_task_success[task] = per_task_success.get(task, 0.0) + success
    per_task_count[task] = per_task_count.get(task, 0) + 1
    item = dict(row)
    item["trial_index"] = index
    item["cumulative_SR"] = cumulative_success / index
    item["task_cumulative_SR"] = per_task_success[task] / per_task_count[task]
    enriched.append(item)
  return enriched


def _average(values: Iterable[float | None]) -> float | None:
  numeric = [value for value in values if value is not None]
  return sum(numeric) / len(numeric) if numeric else None


def _polyline(points: list[tuple[float, float]]) -> str:
  return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _slug(value: str) -> str:
  slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
  return slug or "unknown"


def _task_trial_index(rows: list[dict[str, object]]) -> list[dict[str, object]]:
  counts: dict[str, int] = {}
  enriched = []
  for row in rows:
    task = str(row.get("task_template") or "unknown")
    counts[task] = counts.get(task, 0) + 1
    item = dict(row)
    item["task_trial_index"] = counts[task]
    enriched.append(item)
  return enriched


def _write_svg(
    path: Path,
    *,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
  width, height = 900, 420
  margin_left, margin_right, margin_top, margin_bottom = 70, 24, 52, 58
  colors = [
      "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
      "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
  ]
  all_points = [point for points in series.values() for point in points]
  if not all_points:
    path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    return
  min_x = min(x for x, _ in all_points)
  max_x = max(x for x, _ in all_points)
  min_y = min(y for _, y in all_points) if y_min is None else y_min
  max_y = max(y for _, y in all_points) if y_max is None else y_max
  if max_x == min_x:
    max_x += 1
  if max_y == min_y:
    max_y += 1

  plot_w = width - margin_left - margin_right
  plot_h = height - margin_top - margin_bottom

  def sx(value: float) -> float:
    return margin_left + (value - min_x) / (max_x - min_x) * plot_w

  def sy(value: float) -> float:
    return margin_top + (max_y - value) / (max_y - min_y) * plot_h

  lines = [
      "<svg xmlns='http://www.w3.org/2000/svg' "
      f"width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
      "<rect width='100%' height='100%' fill='white'/>",
      f"<text x='{margin_left}' y='30' font-family='Arial' font-size='20' "
      f"font-weight='700'>{html.escape(title)}</text>",
      f"<line x1='{margin_left}' y1='{height - margin_bottom}' "
      f"x2='{width - margin_right}' y2='{height - margin_bottom}' "
      "stroke='#333'/>",
      f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' "
      f"y2='{height - margin_bottom}' stroke='#333'/>",
      f"<text x='{width / 2:.0f}' y='{height - 16}' text-anchor='middle' "
      "font-family='Arial' font-size='13'>Trial</text>",
      f"<text x='20' y='{height / 2:.0f}' transform='rotate(-90 20 "
      f"{height / 2:.0f})' text-anchor='middle' font-family='Arial' "
      "font-size='13'>Value</text>",
      f"<text x='{margin_left - 8}' y='{sy(min_y):.1f}' text-anchor='end' "
      f"font-family='Arial' font-size='11'>{min_y:.3g}</text>",
      f"<text x='{margin_left - 8}' y='{sy(max_y):.1f}' text-anchor='end' "
      f"font-family='Arial' font-size='11'>{max_y:.3g}</text>",
  ]
  for index, (label, raw_points) in enumerate(series.items()):
    points = [(sx(x), sy(y)) for x, y in raw_points]
    color = colors[index % len(colors)]
    lines.append(
        f"<polyline fill='none' stroke='{color}' stroke-width='2.5' "
        f"points='{_polyline(points)}'/>"
    )
    legend_y = 58 + index * 20
    lines.append(
        f"<line x1='{width - 250}' y1='{legend_y}' x2='{width - 226}' "
        f"y2='{legend_y}' stroke='{color}' stroke-width='3'/>"
    )
    lines.append(
        f"<text x='{width - 220}' y='{legend_y + 4}' font-family='Arial' "
        f"font-size='12'>{html.escape(label)}</text>"
    )
  lines.append("</svg>")
  path.write_text("\n".join(lines), encoding="utf-8")


def _write_timeseries(path: Path, rows: list[dict[str, object]]) -> None:
  fields = [
      "trial_index", "task_trial_index", "task_template", "k", "SA",
      "cumulative_SR", "task_cumulative_SR", "episode_length", "token_total",
      "memory_size",
      "memory_survival_mean", "memory_invalid_click_rate_mean",
      "memory_task_completion_mean", "memory_last_pruned_count",
      "memory_current_capacity", "memory_total_created",
      "memory_total_retrievals", "memory_total_retrieval_hits",
      "memory_total_retrieval_misses", "memory_total_suppressed",
      "memory_total_replays", "memory_total_mutations",
      "memory_total_replacements", "memory_total_pruned",
      "memory_total_deleted_by_risk", "memory_total_embedding_failures",
      "memory_retrieval_hit_rate", "memory_reuse_rate", "memory_mutation_rate",
  ]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in fields})


def _write_task_reports(output_dir: Path, rows: list[dict[str, object]]) -> None:
  task_root = output_dir / "per_task"
  task_root.mkdir(parents=True, exist_ok=True)
  tasks = sorted({str(row.get("task_template") or "unknown") for row in rows})
  index: list[dict[str, object]] = []
  for task in tasks:
    task_rows = [row for row in rows if row.get("task_template") == task]
    task_dir = task_root / _slug(task)
    task_dir.mkdir(parents=True, exist_ok=True)
    _write_timeseries(task_dir / "metrics_timeseries.csv", task_rows)

    sr_values = [_float(row.get("SA")) for row in task_rows]
    step_values = [_float(row.get("episode_length")) for row in task_rows]
    token_values = [_float(row.get("token_total")) for row in task_rows]
    memory_values = [_float(row.get("memory_size")) for row in task_rows]
    half = max(1, len(task_rows) // 2)
    summary = {
        "task_template": task,
        "num_trials": len(task_rows),
        "SR": _average(sr_values),
        "avg_steps_per_trial": _average(step_values),
        "avg_tokens_per_trial": _average(token_values),
        "first_half_avg_steps": _average(step_values[:half]),
        "second_half_avg_steps": _average(step_values[half:]),
        "first_half_avg_tokens": _average(token_values[:half]),
        "second_half_avg_tokens": _average(token_values[half:]),
        "memory_size_initial": next((v for v in memory_values if v is not None), None),
        "memory_size_final": next(
            (v for v in reversed(memory_values) if v is not None), None
        ),
        "memory_size_max": max(
            [v for v in memory_values if v is not None], default=None
        ),
    }
    (task_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    index.append({
        "task_template": task,
        "report_dir": str(task_dir),
        **summary,
    })

    x_values = [float(row["task_trial_index"]) for row in task_rows]
    _write_svg(
        task_dir / "sr_evolution.svg",
        title=f"{task}: SR over 5 Trials",
        series={
            "per-task cumulative SR": [
                (x, float(row["task_cumulative_SR"]))
                for x, row in zip(x_values, task_rows)
            ],
            "trial success": [
                (x, float(_float(row.get("SA")) or 0.0))
                for x, row in zip(x_values, task_rows)
            ],
        },
        y_min=0.0,
        y_max=1.0,
    )

    max_tokens = max([v for v in token_values if v is not None], default=1.0)
    max_steps = max([v for v in step_values if v is not None], default=1.0)
    _write_svg(
        task_dir / "token_steps_trend.svg",
        title=f"{task}: Token and Step Trend",
        series={
            "token_total / task max": [
                (x, value / max_tokens)
                for x, value in zip(x_values, token_values)
                if value is not None
            ],
            "steps / task max": [
                (x, value / max_steps)
                for x, value in zip(x_values, step_values)
                if value is not None
            ],
        },
        y_min=0.0,
        y_max=1.0,
    )

    _write_svg(
        task_dir / "memory_size_curve.svg",
        title=f"{task}: Memory Size",
        series={
            "memory size": [
                (x, value) for x, value in zip(x_values, memory_values)
                if value is not None
            ]
        },
        y_min=0.0,
    )

  (task_root / "task_report_index.json").write_text(
      json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True),
      encoding="utf-8",
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("results_csv", type=Path)
  parser.add_argument("--output_dir", type=Path)
  args = parser.parse_args()

  rows = _task_trial_index(_load_rows(args.results_csv))
  output_dir = args.output_dir or args.results_csv.parent / "dms_metrics_report"
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_timeseries(output_dir / "metrics_timeseries.csv", rows)

  half = max(1, len(rows) // 2)
  step_values = [_float(row.get("episode_length")) for row in rows]
  token_values = [_float(row.get("token_total")) for row in rows]
  memory_values = [_float(row.get("memory_size")) for row in rows]
  summary = {
      "num_trials": len(rows),
      "SR": _average([_float(row.get("SA")) for row in rows]),
      "avg_steps_per_task": _average(step_values),
      "avg_tokens_per_task": _average(token_values),
      "first_half_avg_steps": _average(step_values[:half]),
      "second_half_avg_steps": _average(step_values[half:]),
      "first_half_avg_tokens": _average(token_values[:half]),
      "second_half_avg_tokens": _average(token_values[half:]),
      "memory_size_initial": next((v for v in memory_values if v is not None), None),
      "memory_size_final": next(
          (v for v in reversed(memory_values) if v is not None), None
      ),
      "memory_size_max": max(
          [v for v in memory_values if v is not None], default=None
      ),
  }
  (output_dir / "metrics_summary.json").write_text(
      json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
      encoding="utf-8",
  )

  trial_x = [float(row["trial_index"]) for row in rows]
  _write_svg(
      output_dir / "sr_evolution.svg",
      title="Cumulative Success Rate Evolution",
      series={
          "overall cumulative SR": [
              (x, float(row["cumulative_SR"])) for x, row in zip(trial_x, rows)
          ]
      },
      y_min=0.0,
      y_max=1.0,
  )

  tasks = sorted({str(row.get("task_template") or "unknown") for row in rows})
  task_series: dict[str, list[tuple[float, float]]] = {}
  if len(tasks) <= 10:
    for task in tasks:
      task_series[task] = [
          (float(row["trial_index"]), float(row["task_cumulative_SR"]))
          for row in rows if row.get("task_template") == task
      ]
  else:
    task_series["overall cumulative SR"] = [
        (float(row["trial_index"]), float(row["cumulative_SR"])) for row in rows
    ]
  _write_svg(
      output_dir / "task_sr_evolution.svg",
      title="Per-Task Success Rate Convergence",
      series=task_series,
      y_min=0.0,
      y_max=1.0,
  )

  max_tokens = max([v for v in token_values if v is not None], default=1.0)
  max_steps = max([v for v in step_values if v is not None], default=1.0)
  _write_svg(
      output_dir / "token_steps_trend.svg",
      title="Normalized Token and Step Trend (lower is better)",
      series={
          "token_total / max": [
              (x, (value or 0.0) / max_tokens)
              for x, value in zip(trial_x, token_values)
              if value is not None
          ],
          "steps / max": [
              (x, (value or 0.0) / max_steps)
              for x, value in zip(trial_x, step_values)
              if value is not None
          ],
      },
      y_min=0.0,
      y_max=1.0,
  )

  _write_svg(
      output_dir / "memory_size_curve.svg",
      title="Memory Size Curve",
      series={
          "memory size": [
              (x, value) for x, value in zip(trial_x, memory_values)
              if value is not None
          ]
      },
      y_min=0.0,
  )
  max_memory = max([v for v in memory_values if v is not None], default=1.0)
  if max_memory <= 0:
    max_memory = 1.0
  _write_svg(
      output_dir / "overall_dashboard.svg",
      title="DMS Overall Evolution Dashboard",
      series={
          "cumulative SR": [
              (float(row["trial_index"]), float(row["cumulative_SR"]))
              for row in rows
          ],
          "token_total / max": [
              (x, value / max_tokens)
              for x, value in zip(trial_x, token_values)
              if value is not None
          ],
          "steps / max": [
              (x, value / max_steps)
              for x, value in zip(trial_x, step_values)
              if value is not None
          ],
          "memory size / max": [
              (x, value / max_memory)
              for x, value in zip(trial_x, memory_values)
              if value is not None
          ],
      },
      y_min=0.0,
      y_max=1.0,
  )
  _write_task_reports(output_dir, rows)
  print(f"Wrote DMS metrics report to {output_dir}")


if __name__ == "__main__":
  main()
