#!/usr/bin/env python3
"""Generate all figures from computed results."""

from koide_soddy.figures import (
    fig_scale_sensitivity,
    fig_null_tight_histogram,
    fig_null_loose_breakdown,
)

print("Generating figures...")
fig_scale_sensitivity()
fig_null_tight_histogram()
fig_null_loose_breakdown()
print("Done.")
