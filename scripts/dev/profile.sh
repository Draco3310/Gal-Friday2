#!/bin/bash
set -euo pipefail

source .venv/bin/activate

# Profile the application
python -m cProfile -o profile.stats -m gal_friday.main --config config/config.yaml "$@"

# Generate report
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.strip_dirs()
p.sort_stats('cumulative')
p.print_stats(50)
" > profile_report.txt

echo "Profile report saved to profile_report.txt"

# Visualize if snakeviz is installed
if command -v snakeviz &>/dev/null; then
    snakeviz profile.stats
fi
