"""Figure generation for Koide-Soddy analysis."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fig_scale_sensitivity(scale_json: str = "results/scale_sensitivity.json",
                          output: str = "results/fig_scale_sensitivity.pdf"):
    """m_s(mu) vs mu with F^2 horizontal line and 1-sigma band."""
    with open(scale_json) as f:
        data = json.load(f)

    rows = data["data"]
    mu = np.array([r["mu_mev"] for r in rows])
    ms = np.array([r["ms_mu_mev"] for r in rows])
    sigma = np.array([r["ms_uncertainty_mev"] for r in rows])
    F2 = data["F_squared_mev"]
    mu_star = data["mu_star_mev"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mu / 1000, ms, 'b-', linewidth=1.5, label=r'$m_s(\mu)$')
    ax.fill_between(mu / 1000, ms - sigma, ms + sigma, alpha=0.2, color='b',
                    label=r'$\pm 1\sigma$ (lattice)')
    ax.axhline(F2, color='r', linestyle='--', linewidth=1.5,
               label=rf'$\mathcal{{F}}^2 = {F2:.1f}$ MeV')
    ax.axvline(mu_star / 1000, color='gray', linestyle=':', linewidth=1,
               label=rf'$\mu = \Sigma m_\ell = {mu_star/1000:.2f}$ GeV')

    ax.set_xlabel(r'$\mu$ [GeV]', fontsize=12)
    ax.set_ylabel(r'Mass [MeV]', fontsize=12)
    ax.set_title(r'Scale dependence of $m_s(\mu)$ vs. $\mathcal{F}^2$', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(mu.min() / 1000, mu.max() / 1000)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved {output} + .png")


def fig_null_tight_histogram(residuals_npy: str = "results/null_tight_residuals.npy",
                             tight_json: str = "results/null_tight.json",
                             output: str = "results/fig_null_tight_histogram.pdf"):
    """Histogram of F^2 - m_s(mu*) in sigma units for null triples."""
    residuals = np.load(residuals_npy)
    with open(tight_json) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Clip for display
    clip = 50
    r_clip = residuals[(residuals > -clip) & (residuals < clip)]

    ax.hist(r_clip, bins=100, density=True, alpha=0.7, color='steelblue',
            edgecolor='none')
    ax.axvline(0, color='red', linestyle='--', linewidth=2,
               label=r'Observed ($\mathcal{F}^2 \approx m_s$)')
    ax.axvspan(-1, 1, alpha=0.1, color='green', label=r'$\pm 1\sigma$ window')

    hit_frac = data["hit_fraction"]
    n_test = data["n_valid_triples"]
    ax.set_xlabel(r'$(\mathcal{F}^2 - m_s(\mu_\star)) / \sigma_{m_s}$', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'Tight null: {100*hit_frac:.2f}% hit rate ({n_test:,} triples tested)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-clip, clip)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved {output} + .png")


def fig_null_loose_breakdown(loose_json: str = "results/null_loose.json",
                             output: str = "results/fig_null_loose_breakdown.pdf"):
    """Bar chart of per-quark hit rates from the loose null."""
    with open(loose_json) as f:
        data = json.load(f)

    quarks = list(data["per_quark_hits"].keys())
    hits = [data["per_quark_hits"][q] for q in quarks]
    n_test = data["n_valid_triples"]
    power_n = data.get("power_n", 2)

    # Convert to fractions (%)
    hit_pct = [100.0 * h / n_test for h in hits]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(quarks))
    ax.bar(x, hit_pct, 0.5,
           label=rf'$\mathcal{{F}}^{{{power_n}}}$', alpha=0.8)

    ax.set_xlabel('Quark', fontsize=12)
    ax.set_ylabel('Hit rate (%)', fontsize=12)
    ax.set_title(rf'Loose null: per-quark $\mathcal{{F}}^{{{power_n}}}$ hit rates ({n_test:,} triples)',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(quarks, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output)
    fig.savefig(output.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved {output} + .png")
