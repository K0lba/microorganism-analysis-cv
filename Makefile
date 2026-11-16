# Обёртка над tasks.ps1. На этой машине make не установлен:
# запускайте  .\tasks.ps1 help
MIN_AREA ?= 20
OBJECTS ?= auto
OUT ?= results/morphology.csv
MODEL ?= classical

.PHONY: help test synthetic demo benchmark compare schema video morphology

help test synthetic demo benchmark compare schema robustness feature_errors plots:
	powershell -NoProfile -ExecutionPolicy Bypass -File tasks.ps1 $@ -Model "$(MODEL)"

video:
	powershell -NoProfile -ExecutionPolicy Bypass -File tasks.ps1 video -Source "$(SOURCE)" -Model "$(MODEL)"

morphology:
	powershell -NoProfile -ExecutionPolicy Bypass -File tasks.ps1 morphology -Image "$(IMAGE)" -Model "$(MODEL)" -MinArea "$(MIN_AREA)" -Objects "$(OBJECTS)" -Out "$(OUT)"
