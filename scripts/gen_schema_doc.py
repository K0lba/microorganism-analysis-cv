#!/usr/bin/env python
"""Сгенерировать docs/export_schema.md из схемы таблицы (microtrack.export.COLUMNS).

Запуск после любого изменения COLUMNS:
    python scripts/gen_schema_doc.py
tests/test_export.py проверяет, что файл совпадает со схемой.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microtrack.export import schema_markdown  # noqa: E402


def main() -> int:
    path = ROOT / "docs" / "export_schema.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema_markdown(), encoding="utf-8", newline="\n")
    print(f"записано: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
