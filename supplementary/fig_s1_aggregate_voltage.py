#!/usr/bin/env python3
"""
fig_s1_aggregate_voltage.py
────────────────────────────
Monte-Carlo plausibility check: aggregate RMS voltage from ~10⁴
microtubules with independent or partially correlated phases.

Model
─────
V(t) = Σᵢ aᵢ mᵢ A₀ sin(ωt + φᵢ)

  A₀    = √2 · V₀_rms          (peak amplitude per MT)
  V₀_rms = 0.63 µV              (Cantero & Cantiello 2023, verified)
  φᵢ    ~ von Mises(0, κ)       (κ=0 → uniform; κ→∞ → coherent)
  mᵢ    ~ Bernoulli(p)          (active MT fraction)
  aᵢ    ~ LogNormal(0, σ_a)     (amplitude heterogeneity)

Output
──────
  results/fig_s1_aggregate_voltage.pdf / .png
  Panel A: V_rms vs κ  (phase coherence sweep)
  Panel B: V_rms vs N  (MT count scaling)
  Panel C: V_rms vs p  (active fraction sweep)
  Panel D: parameter-space heatmap f(κ, p) → V_rms

Author : Emre Özyürt
Date   : March 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import vonmises

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
V0_RMS    = 0.63e-6       # V  (per-MT RMS, verified from Cantero 2023)
A0        = np.sqrt(2) * V0_RMS   # peak amplitude
F_MT      = 39.0          # Hz
OMEGA     = 2 * np.pi * F_MT
N_REF     = 10_000        # reference MT count (layer-V pyramidal)
N_TRIALS  = 500          # Monte-Carlo realisations per condition
SEED      = 20260310
SIGMA_A   = 0.3           # log-normal spread for amplitude heterogeneity

# physiological reference voltages (V)
V_MEPSP       = 0.5e-3   # typical miniature EPSP
V_AP_THRESH   = 20.0e-3  # somatic AP threshold
V_ION_CHAN_LO = 1.0e-3   # ion channel modulation range, low
V_ION_CHAN_HI = 10.0e-3  # ion channel modulation range, high

# ══════════════════════════════════════════════════════════════════
#  CORE SIMULATION
# ══════════════════════════════════════════════════════════════════
def simulate_aggregate(N, p, kappa, sigma_a, n_trials, rng):
    """
    Compute aggregate V_rms over n_trials realisations.

    For each trial:
      - draw N phases from von Mises(0, κ)
      - draw N active masks from Bernoulli(p)
      - draw N amplitude factors from LogNormal(0, σ_a)
      - compute phasor sum → V_peak → V_rms = V_peak / √2

    Returns array of shape (n_trials,) with V_rms values in volts.
    """
    vrms_all = np.empty(n_trials)

    # batch in chunks to manage memory
    chunk = min(n_trials, 5000)
    idx = 0
    while idx < n_trials:
        n = min(chunk, n_trials - idx)

        # phases: (n, N) — von Mises(0, κ)
        if kappa < 1e-6:
            phases = rng.uniform(0, 2 * np.pi, size=(n, N))
        else:
            phases = vonmises.rvs(kappa, loc=0, size=(n, N),
                                  random_state=rng)

        # active mask: (n, N)
        mask = rng.random(size=(n, N)) < p

        # amplitude factors: (n, N)
        a = rng.lognormal(mean=0.0, sigma=sigma_a, size=(n, N))
        # normalise so E[a] = 1
        a = a / np.exp(0.5 * sigma_a**2)

        # phasor sum: V_peak = |Σ aᵢ mᵢ A₀ exp(jφᵢ)|
        phasors = a * mask * A0 * np.exp(1j * phases)  # (n, N)
        resultant = np.abs(phasors.sum(axis=1))         # (n,)

        # V_rms = V_peak / √2
        vrms_all[idx:idx+n] = resultant / np.sqrt(2)
        idx += n

    return vrms_all


def analytical_incoherent_rms(N, p, sigma_a):
    """
    Analytical V_rms for fully incoherent (κ=0) case.
    E[|Σ|²] = N·p · E[a²] · A₀²
    V_rms = √(N·p · E[a²]) · A₀ / √2
          = √(N·p · E[a²]) · V₀_RMS
    For LogNormal(0,σ): E[a²]/E[a]² = exp(σ²), and after
    normalisation E[a]=1, so E[a²] = exp(σ²).
    """
    Ea2 = np.exp(sigma_a**2)
    return np.sqrt(N * p * Ea2) * V0_RMS


# ══════════════════════════════════════════════════════════════════
#  PANEL A: V_rms vs κ  (phase coherence sweep)
# ══════════════════════════════════════════════════════════════════
def panel_a(ax, rng):
    kappas = np.concatenate([
        np.linspace(0, 1, 20),
        np.linspace(1, 10, 20),
        np.linspace(10, 100, 15)
    ])
    kappas = np.unique(kappas)

    medians = []
    ci_lo = []
    ci_hi = []

    for k in kappas:
        v = simulate_aggregate(N_REF, 1.0, k, SIGMA_A, max(100, N_TRIALS//2), rng)
        v_uV = v * 1e6  # to µV
        medians.append(np.median(v_uV))
        ci_lo.append(np.percentile(v_uV, 2.5))
        ci_hi.append(np.percentile(v_uV, 97.5))

    medians = np.array(medians)
    ci_lo = np.array(ci_lo)
    ci_hi = np.array(ci_hi)

    ax.fill_between(kappas, ci_lo, ci_hi, alpha=0.25, color='C0')
    ax.plot(kappas, medians, 'C0-', lw=2, label='Median')
    ax.axhline(analytical_incoherent_rms(N_REF, 1.0, SIGMA_A) * 1e6,
               ls='--', color='grey', lw=1, label='Analytic (κ=0)')

    # physiological references
    ax.axhspan(V_ION_CHAN_LO * 1e6, V_ION_CHAN_HI * 1e6,
               alpha=0.08, color='green', label='Ion ch. mod. range')
    ax.axhline(V_MEPSP * 1e6, ls=':', color='red', lw=1,
               label=f'mEPSP ({V_MEPSP*1e3:.1f} mV)')

    ax.set_xlabel('Phase concentration κ')
    ax.set_ylabel('V_rms (µV)')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_yscale('log')
    ax.set_title('A) Phase coherence sweep (N=10⁴, p=1)')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(0, 100)


# ══════════════════════════════════════════════════════════════════
#  PANEL B: V_rms vs N  (MT count scaling)
# ══════════════════════════════════════════════════════════════════
def panel_b(ax, rng):
    Ns = np.logspace(1, 5, 30).astype(int)
    Ns = np.unique(Ns)

    for kappa, color, label in [(0, 'C0', 'Incoherent (κ=0)'),
                                 (5, 'C1', 'Partial (κ=5)'),
                                 (50, 'C2', 'High coh. (κ=50)')]:
        meds = []
        for N in Ns:
            v = simulate_aggregate(N, 1.0, kappa, SIGMA_A,
                                   max(50, N_TRIALS//5), rng)
            meds.append(np.median(v * 1e6))
        ax.plot(Ns, meds, '-o', color=color, ms=3, lw=1.5, label=label)

    # analytic √N line
    Ns_cont = np.logspace(1, 5, 200)
    ax.plot(Ns_cont,
            analytical_incoherent_rms(Ns_cont, 1.0, SIGMA_A) * 1e6,
            'k--', lw=1, alpha=0.5, label='∝ √N (analytic)')

    ax.axhline(V_MEPSP * 1e6, ls=':', color='red', lw=1)
    ax.axhspan(V_ION_CHAN_LO * 1e6, V_ION_CHAN_HI * 1e6,
               alpha=0.08, color='green')

    ax.set_xlabel('Number of microtubules N')
    ax.set_ylabel('V_rms (µV)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('B) MT count scaling (p=1)')
    ax.legend(fontsize=7)


# ══════════════════════════════════════════════════════════════════
#  PANEL C: V_rms vs p  (active fraction sweep)
# ══════════════════════════════════════════════════════════════════
def panel_c(ax, rng):
    ps = np.linspace(0.05, 1.0, 25)

    for kappa, color, label in [(0, 'C0', 'κ=0'),
                                 (5, 'C1', 'κ=5'),
                                 (20, 'C2', 'κ=20')]:
        meds = []
        ci_lo = []
        ci_hi = []
        for p in ps:
            v = simulate_aggregate(N_REF, p, kappa, SIGMA_A,
                                   max(50, N_TRIALS//5), rng)
            v_uV = v * 1e6
            meds.append(np.median(v_uV))
            ci_lo.append(np.percentile(v_uV, 2.5))
            ci_hi.append(np.percentile(v_uV, 97.5))

        meds = np.array(meds)
        ci_lo = np.array(ci_lo)
        ci_hi = np.array(ci_hi)
        ax.fill_between(ps, ci_lo, ci_hi, alpha=0.15, color=color)
        ax.plot(ps, meds, '-', color=color, lw=2, label=label)

    ax.axhline(V_MEPSP * 1e6, ls=':', color='red', lw=1)
    ax.set_xlabel('Active MT fraction p')
    ax.set_ylabel('V_rms (µV)')
    ax.set_title('C) Active fraction sweep (N=10⁴)')
    ax.legend(fontsize=7)


# ══════════════════════════════════════════════════════════════════
#  PANEL D: Heatmap f(κ, p) → V_rms
# ══════════════════════════════════════════════════════════════════
def panel_d(ax, rng):
    kappas = np.linspace(0, 30, 40)
    ps = np.linspace(0.1, 1.0, 30)
    grid = np.empty((len(ps), len(kappas)))

    for i, p in enumerate(ps):
        for j, k in enumerate(kappas):
            v = simulate_aggregate(N_REF, p, k, SIGMA_A, 50, rng)
            grid[i, j] = np.median(v * 1e6)

    im = ax.pcolormesh(kappas, ps, grid, shading='auto',
                       cmap='viridis', norm=LogNorm())
    plt.colorbar(im, ax=ax, label='V_rms (µV)')

    # contour lines at physiological thresholds
    cs = ax.contour(kappas, ps, grid,
                    levels=[63, 100, 500, 1000, 5000],
                    colors='white', linewidths=0.8)
    ax.clabel(cs, fmt='%d µV', fontsize=6, colors='white')

    ax.set_xlabel('Phase concentration κ')
    ax.set_ylabel('Active MT fraction p')
    ax.set_title('D) Parameter space → V_rms (N=10⁴)')


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Supplementary Figure S1: Microtubule Aggregate Voltage '
        'Monte Carlo Analysis\n'
        f'V₀(rms) = {V0_RMS*1e6:.2f} µV per MT  |  '
        f'{N_TRIALS:,} trials  |  seed = {SEED}',
        fontsize=11, fontweight='bold'
    )

    panel_a(axes[0, 0], rng)
    panel_b(axes[0, 1], rng)
    panel_c(axes[1, 0], rng)
    panel_d(axes[1, 1], rng)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # ── summary statistics printed to console ──
    print('='*60)
    print('SUMMARY — MT Aggregate Voltage Monte Carlo')
    print('='*60)

    v_ref = simulate_aggregate(N_REF, 1.0, 0, SIGMA_A, 2000, rng)
    v_uV = v_ref * 1e6
    print(f'\nReference case: N={N_REF}, p=1.0, κ=0 (incoherent)')
    print(f'  Analytic V_rms  = '
          f'{analytical_incoherent_rms(N_REF, 1.0, SIGMA_A)*1e6:.2f} µV')
    print(f'  MC median       = {np.median(v_uV):.2f} µV')
    print(f'  MC mean         = {np.mean(v_uV):.2f} µV')
    print(f'  MC 2.5–97.5% CI = [{np.percentile(v_uV,2.5):.2f}, '
          f'{np.percentile(v_uV,97.5):.2f}] µV')
    print(f'  MC V_peak med   = {np.median(v_uV)*np.sqrt(2):.2f} µV')
    print(f'\n  vs mEPSP ({V_MEPSP*1e6:.0f} µV): '
          f'{np.median(v_uV)/V_MEPSP/1e6*100:.1f}%')
    print(f'  vs AP threshold ({V_AP_THRESH*1e3:.0f} mV): '
          f'{np.median(v_uV)/V_AP_THRESH/1e6*100:.3f}%')

    # bundle amplification scenario
    BUNDLE_FACTOR = 300
    v_bundle = v_uV * BUNDLE_FACTOR
    print(f'\nBundle amplification scenario (×{BUNDLE_FACTOR}):')
    print(f'  Effective V₀_rms = {V0_RMS*1e6*BUNDLE_FACTOR:.1f} µV')
    print(f'  Aggregate median = {np.median(v_bundle):.0f} µV '
          f'= {np.median(v_bundle)/1e3:.1f} mV')
    print(f'  Aggregate 95% CI = [{np.percentile(v_bundle,2.5):.0f}, '
          f'{np.percentile(v_bundle,97.5):.0f}] µV')

    print('\n' + '='*60)

    # Save to RESULTS_DIR instead of current directory
    out_pdf = os.path.join(RESULTS_DIR, 'fig_s1_aggregate_voltage.pdf')
    out_png = os.path.join(RESULTS_DIR, 'fig_s1_aggregate_voltage.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'\nFigures saved: {out_pdf} / .png')
    plt.close() # replaced plt.show()


if __name__ == '__main__':
    main()
