# Koide-Soddy Analysis

This codebase exists to test whether an empirical observation about the Koide lepton triple and the strange-quark mass is structurally meaningful or a chance match. Both outcomes are valid; the null-model results in `results/null_tight.json` and `results/null_loose.json` are the answer. See the headline `hit_fraction` keys.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+ and `rundec` (installed automatically). Use `python3.13` if your default Python doesn't have rundec.

## Run

```bash
# Run tests first
python3.13 -m pytest tests/ -v

# Run all scripts in order
python3.13 scripts/01_verify_central_claim.py
python3.13 scripts/02_scale_sensitivity.py
python3.13 scripts/03_input_sensitivity.py
python3.13 scripts/04_null_model_tight.py 2>/dev/null
python3.13 scripts/05_null_model_loose.py 2>/dev/null
python3.13 scripts/06_make_figures.py
```

The `2>/dev/null` suppresses verbose CRunDec warnings at low scales (these are harmless).

## Output files

| File | Description |
|---|---|
| `results/scale_sensitivity.json` | m_s(mu) vs mu, with 1-sigma and 2-sigma windows |
| `results/input_sensitivity.json` | Full error budget on F^2 - m_s |
| `results/null_tight.json` | **Headline result**: hit fraction for F^2 = m_s(mu*) |
| `results/null_loose.json` | Data-dredging check: any F^n matching any quark |
| `results/null_tight_residuals.npy` | Raw residuals for histogram |

## Figures

- [Scale sensitivity](results/fig_scale_sensitivity.png) (`results/fig_scale_sensitivity.pdf`)
- [Tight null histogram](results/fig_null_tight_histogram.png) (`results/fig_null_tight_histogram.pdf`)
- [Loose null breakdown](results/fig_null_loose_breakdown.png) (`results/fig_null_loose_breakdown.pdf`)

## Constants

- Lepton masses: PDG 2024 pole masses
- Light quark masses: FLAG 2024 N_f = 2+1+1 averages at 2 GeV
- Heavy quark masses: PDG 2024 at self-scales
- alpha_s(M_Z) = 0.1180 (PDG 2024)
- QCD running: four-loop via rundec with threshold matching at m_c and m_b

## Reproducibility

- Random seed: 42 (null model samplers)
- Null model uses 100,000 Koide triples
- Perturbative regime: only triples with mu* in [1, 50] GeV are tested (25.5% of total)
- Runtime: < 5 seconds total on Apple M-series
- Libraries: rundec 0.6, numpy, scipy, matplotlib, Python 3.13
