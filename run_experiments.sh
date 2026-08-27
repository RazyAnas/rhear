#!/usr/bin/env bash
# Run the RHEAR digital twin. No hardware, no datasets, no downloads.
set -e
cd "$(dirname "$0")"
mkdir -p results
echo "=== E00  simulator validation ==="   && python3 rhear/experiments/e00_validate.py
echo && echo "=== E01  causality / codec ===" && python3 rhear/experiments/e01_causality.py
echo && echo "=== E02  impulsive noise ==="   && python3 rhear/experiments/e02_impulsive.py
echo && echo "results written to results/"
