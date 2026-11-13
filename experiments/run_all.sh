#!/bin/sh
# Runs E2-E8 sequentially; each writes results/<name>.json and a log.
cd "$(dirname "$0")"
for e in e3_rf_noise e2_scale e5_importance e7_reliability e6_timing e4_placement e8_powergraph; do
  echo "=== $e $(date +%T)"
  python -u $e.py > results/${e}_log.txt 2>&1 || echo "FAILED $e"
done
echo "=== done $(date +%T)"
