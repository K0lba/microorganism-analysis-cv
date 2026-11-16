#!/usr/bin/env python
"""Отчёт об ошибке каждого признака относительно истинных размеров из objects.csv.

Сопоставляет измеренные признаки (из таблицы наблюдений конвейера) с эталонными
(из objects.csv синтетики) и считает среднюю и медианную относительную ошибку
каждого признака. Это главный пункт проверки плана: модель может хорошо
находить бактерии, но периметр при этом считать неточно.

Запуск:
    python scripts/feature_errors.py --table results/demo/observations.csv --gt data/synthetic_demo/objects.csv
    python scripts/feature_errors.py --table results/demo/observations.csv --gt data/synthetic_demo/objects.csv --out results/feature_errors

Результаты (папка --out, по умолчанию рядом с таблицей):
    feature_errors.csv   по каждому признаку: средняя и медианная ошибка, bias, RMSE
    feature_errors.md    таблица для отчёта
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microtrack.evaluation import compute_errors, load_gt, match_tracks  # noqa: E402


def build_markdown(errors: pd.DataFrame) -> str:
    lines = ["# Ошибка признаков относительно эталона", "",
             "Относительная ошибка = (измерено − эталон) / эталон. "
             "Положительный bias — систематическое завышение.", ""]
    cols = ["feature", "n", "mean_rel_error", "median_rel_error", "mean_abs_rel_error",
            "bias", "rmse", "gt_mean", "pred_mean"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in errors.iterrows():
        lines.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    lines += ["", "## Что смотреть", "",
              "- **mean_rel_error** ~ 0 — нет систематического смещения.",
              "- **mean_abs_rel_error** — средняя по модулю ошибка (как в плане: «AR ошибается на 3 %»).",
              "- **bias** — положительный = систематическое завышение, отрицательный = занижение.",
              "- Если в эталоне уже есть площадь и периметр маски, берётся именно они.",
              "- Иначе площадь и периметр восстанавливаются из length/width как у эллипса."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="observations.csv из конвейера")
    ap.add_argument("--gt", required=True, help="objects.csv из синтетики")
    ap.add_argument("--out", help="каталог вывода (по умолчанию рядом с таблицей)")
    args = ap.parse_args(argv)

    table = pd.read_csv(args.table)
    gt = load_gt(Path(args.gt))
    matched = match_tracks(table, gt)
    errors = compute_errors(matched, gt)
    out = Path(args.out) if args.out else Path(args.table).parent
    out.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out / "feature_errors.csv", index=False)
    (out / "feature_errors.md").write_text(build_markdown(errors), encoding="utf-8")
    print(build_markdown(errors))
    print(f"\nСопоставлено {len(matched)} из {len(table)} строк; сохранено: {(out / 'feature_errors.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
