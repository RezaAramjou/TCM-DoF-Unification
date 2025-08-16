# run_analysis_v20.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v20_definitive import CmaSolverDefinitive 
import matplotlib.pyplot as plt
from scipy.special import lpmv
import warnings

def complex_interp(x_new, x_old, y_old_complex):
    """Properly interpolates complex data."""
    y_real = np.interp(x_new, x_old, np.real(y_old_complex))
    y_imag = np.interp(x_new, x_old, np.imag(y_old_complex))
    return y_real + 1j * y_imag

def p_lm_stable(l, m, x):
    """Computes associated Legendre polynomials P_l^m(x) using stable recurrence relations."""
    if m < 0 or m > l: return np.zeros_like(x)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt((1.0-x)*(1.0+x))
        fact = 1.0
        for i in range(1, m + 1): pmm = -pmm * fact * somx2; fact += 2.0
    if l == m: return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1: return pmmp1
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1; pmmp1 = pll
    return pll

def fit_spherical_waves(theta, E_pattern_complex, max_N, error_threshold):
    """Robustly fits a complex pattern using stabilized Legendre polynomials."""
    n_gauss_points = 512
    costh_nodes, weights = np.polynomial.legendre.leggauss(n_gauss_points)
    theta_gauss = np.arccos(costh_nodes)
    E_pattern_interp = complex_interp(theta_gauss, theta, E_pattern_complex)

    for N in range(1, max_N + 1):
        A = np.zeros((n_gauss_points, N), dtype=np.float64)
        for n in range(1, N + 1):
            A[:, n - 1] = np.sqrt((2 * n + 1) / (2 * n * (n + 1))) * p_lm_stable(n, 1, costh_nodes)
        Aw = A * np.sqrt(weights)[:, np.newaxis]
        if np.linalg.cond(Aw) > 1e12: warnings.warn(f'Fit matrix is ill-conditioned for N={N}.', UserWarning)
        try:
            x, _, _, _ = np.linalg.lstsq(Aw, E_pattern_interp * np.sqrt(weights), rcond=None)
            E_reconstructed = A @ x
            power_true = np.sum(weights * np.abs(E_pattern_interp)**2)
            power_err = np.sum(weights * np.abs(E_pattern_interp - E_reconstructed)**2)
            if power_true > 1e-20 and (power_err / power_true) < error_threshold: return N
        except np.linalg.LinAlgError: pass
    warnings.warn('Fit did not converge. Returning max_N.', UserWarning)
    return max_N

def run_analysis_for_length(L, f0, a_const, segments_per_lambda, feed_gap):
    """Helper function for parallel execution with controlled meshing."""
    lambda0 = 299792458.0 / f0
    num_segments = int(L / lambda0 * segments_per_lambda)
    if num_segments % 2 == 0: num_segments += 1
    if num_segments < 21: num_segments = 21

    config = {
        'Dipole': {'Length': L, 'Radius': a_const, 'FeedGap': feed_gap},
        'Mesh': {'Segments': num_segments},
        'Numerics': {'Accuracy': 'medium'},
        'Execution': {'Frequency': f0, 'NumModes': 10, 'Verbose': False}
    }
    solver = CmaSolverDefinitive(config)
    return solver.run()

def run_benchmark(a_const, segments_per_lambda, feed_gap):
    """Performs the canonical half-wave dipole benchmark."""
    print("\n--- Running Canonical Half-Wave Dipole Benchmark ---")
    # Sweep length around L=0.5*lambda to find resonance
    L_sweep = np.linspace(0.45, 0.5, 51) * (299792458.0 / 300e6)
    Z_in_list = []
    for L in tqdm(L_sweep, desc="Benchmark Sweep"):
        res = run_analysis_for_length(L, 300e6, a_const, segments_per_lambda, feed_gap)
        Z_in_list.append(res['InputImpedance'])
    
    Z_in_list = np.array(Z_in_list)
    # Find resonance (zero-crossing of reactance)
    resonant_idx = np.argmin(np.abs(np.imag(Z_in_list)))
    resonant_L = L_sweep[resonant_idx]
    resonant_Zin = Z_in_list[resonant_idx]
    
    print(f"Benchmark Complete:")
    print(f"  - Resonant Length Found: {resonant_L:.4f} m (~{resonant_L / (299792458.0/300e6):.3f} lambda)")
    print(f"  - Input Impedance at Resonance: {resonant_Zin.real:.2f} + {resonant_Zin.imag:.2f}j Ohms")
    print(f"  - Canonical Theoretical Value: ~73 + 0j Ohms")
    
    # Plotting the benchmark result
    plt.figure(figsize=(8, 6))
    plt.plot(L_sweep, np.real(Z_in_list), '-o', label='Input Resistance (R_in)')
    plt.plot(L_sweep, np.imag(Z_in_list), '--s', label='Input Reactance (X_in)')
    plt.axvline(resonant_L, color='r', linestyle='--', label=f'Resonance at L={resonant_L:.3f}m')
    plt.title('Benchmark: Half-Wave Dipole Input Impedance', fontweight='bold')
    plt.xlabel('Dipole Length (m)'); plt.ylabel('Impedance (Ohms)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig('Fig_Benchmark_v20.png', dpi=300)
    print("Benchmark plot saved to Fig_Benchmark_v20.png")
    plt.show()


# --- Step 1: Define Sweep Parameters ---
print('--- Unification & Q-Factor Analysis (v20.0 Definitive Edition) ---')
f0 = 300e6
lambda0 = 299792458.0 / f0
a_const = 0.001 * lambda0
segments_per_lambda = 50 # Increased density for higher accuracy
feed_gap = 0.01 * lambda0
L_over_lambda_sweep = np.linspace(0.2, 1.5, 27) # Refined sweep range
Ls = L_over_lambda_sweep * lambda0

# --- Step 2: Run Benchmark First ---
run_benchmark(a_const, segments_per_lambda, feed_gap)

# --- Step 3: Run Main Analysis Sweep ---
print('\n--- Starting Main Analysis Sweep ---')
results = Parallel(n_jobs=-1)(
    delayed(run_analysis_for_length)(L, f0, a_const, segments_per_lambda, feed_gap)
    for L in tqdm(Ls, desc='Overall Progress')
)
print('Main analysis sweep complete.')

# --- Step 4: Post-Process and Analyze Results ---
print('Performing final DoF and Q-Factor analysis...')
NDoF_TCM = [np.sum(np.abs(r['lambda_n']) < 1.0) for r in results]
Q_mode1, Q_mode2 = [], []
N_Yaghjian_mode1 = []

for r in tqdm(results, desc="Analyzing Results"):
    q_sorted = r['Q_n'][np.argsort(np.abs(r['lambda_n']))]
    Q_mode1.append(q_sorted[0])
    Q_mode2.append(q_sorted[1] if len(q_sorted) > 1 else np.nan)
        
    theta = np.linspace(1e-6, np.pi - 1e-6, 361)
    J_mode1 = r['J_n'][:, np.argsort(np.abs(r['lambda_n']))[0]][:, np.newaxis]
    E_theta_mode1 = CmaSolverDefinitive.calculate_radiation_properties(
        {'J_n': J_mode1, **r}, theta_rad=theta
    )
    
    if np.max(np.abs(E_theta_mode1)) > 1e-12:
        N_fit = fit_spherical_waves(theta, E_theta_mode1[0,:], 20, 0.05)
        N_Yaghjian_mode1.append(N_fit)
    else:
        N_Yaghjian_mode1.append(0)

# --- Step 5: Plotting ---
print('Generating final report plots...')
plt.style.use('default')
styles = CmaSolverDefinitive.get_plot_styles()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('CMA Unification & Q-Factor Analysis (v20.0 Definitive Edition)', fontsize=16, fontweight='bold')

ax1.plot(L_over_lambda_sweep, NDoF_TCM, '-o', color=styles['Color1'], label='Intrinsic DoF from Eigenvalues (|λn| < 1)')
ax1.plot(L_over_lambda_sweep, N_Yaghjian_mode1, '--s', color=styles['Color2'], label='Intrinsic DoF from Far-Field of Mode 1 (N_fit)')
ax1.set_xlabel('Dipole Length (L/λ)', fontweight='bold'); ax1.set_ylabel('Degrees of Freedom (NDoF)')
ax1.set_title('Unification of Intrinsic Degrees of Freedom', fontweight='bold')
ax1.legend(); ax1.grid(True)

ax2.plot(L_over_lambda_sweep, Q_mode1, '-o', color=styles['Color1'], label='Q-Factor of Mode 1')
ax2.plot(L_over_lambda_sweep, Q_mode2, '--s', color=styles['Color2'], label='Q-Factor of Mode 2')
ax2.set_xlabel('Dipole Length (L/λ)', fontweight='bold'); ax2.set_ylabel('Energy-Based Quality Factor (Q)')
ax2.set_title('Modal Q-Factor vs. Dipole Length (Controlled Mesh)', fontweight='bold')
ax2.set_yscale('log'); ax2.legend(); ax2.grid(True, which="both", ls="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Fig_Report_v20_Definitive.png', dpi=300)

print('Definitive report saved to Fig_Report_v20_Definitive.png. Analysis is complete.')
plt.show()
