#!/usr/bin/env bash
# Run the RHEAR digital twin. No hardware, no datasets, no downloads.
set -e
cd "$(dirname "$0")"
mkdir -p results
echo "=== E00  simulator validation ==="   && python3 rhear/experiments/e00_validate.py
echo && echo "=== E01  causality / codec ===" && python3 rhear/experiments/e01_causality.py
echo && echo "=== E02  impulsive noise ==="   && python3 rhear/experiments/e02_impulsive.py
echo && echo "=== G5 tests ===" && python3 tests/test_geometry.py && python3 tests/test_doa.py
echo && echo "=== E05  direction ===" && python3 rhear/experiments/e05_direction.py
echo && echo "=== E06  neural filter selection ===" && python3 rhear/experiments/e06_filter_generation.py
echo && echo "results written to results/"
