# run_analysis_v18.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v18_definitive import CmaSolverDefinitive 
import matplotlib.pyplot as plt
from scipy.special import lpmv
import warnings

def complex_interp(x_new, x_old, y_old_complex):
    """Properly interpolates complex data by handling real and imaginary parts separately."""
    y_real = np.interp(x_new, x_old, np.real(y_old_complex))
    y_imag = np.interp(x_new, x_old, np.imag(y_old_complex))
    return y_real + 1j * y_imag

def p_lm_stable(l, m, x):
    """
    Computes associated Legendre polynomials P_l^m(x) using stable
    three-term recurrence relations to avoid numerical issues for large l.
    """
    if m < 0 or m > l: return np.zeros_like(x)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt((1.0-x)*(1.0+x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = -pmm * fact * somx2
            fact += 2.0
    if l == m: return pmm
    
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1: return pmmp1
    
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def fit_spherical_waves(theta, E_pattern_complex, max_N, error_threshold):
    """
    Robustly fits a complex pattern using stabilized Legendre polynomials
    and includes a condition number check.
    """
    n_gauss_points = 512
    costh_nodes, weights = np.polynomial.legendre.leggauss(n_gauss_points)
    theta_gauss = np.arccos(costh_nodes)
    E_pattern_interp = complex_interp(theta_gauss, theta, E_pattern_complex)

    for N in range(1, max_N + 1):
        A = np.zeros((n_gauss_points, N), dtype=np.float64)
        for n in range(1, N + 1):
            norm_factor = np.sqrt((2 * n + 1) / (2 * n * (n + 1)))
            A[:, n - 1] = norm_factor * p_lm_stable(n, 1, costh_nodes)

        Aw = A * np.sqrt(weights)[:, np.newaxis]
        if np.linalg.cond(Aw) > 1e12:
            warnings.warn(f'Fit matrix is ill-conditioned for N={N}.', UserWarning)

        try:
            x, _, _, _ = np.linalg.lstsq(Aw, E_pattern_interp * np.sqrt(weights), rcond=None)
            E_reconstructed = A @ x
            power_true = np.sum(weights * np.abs(E_pattern_interp)**2)
            power_err = np.sum(weights * np.abs(E_pattern_interp - E_reconstructed)**2)
            if power_true > 1e-20 and (power_err / power_true) < error_threshold:
                return N
        except np.linalg.LinAlgError: pass
    
    warnings.warn('Fit did not converge. Returning max_N.', UserWarning)
    return max_N

def run_analysis_for_length(L, f0, a_const, N_const, feed_gap):
    """Helper function for parallel execution."""
    config = {
        'Dipole': {'Length': L, 'Radius': a_const, 'FeedGap': feed_gap},
        'Mesh': {'Segments': N_const},
        'Numerics': {'Accuracy': 'medium'},
        'Execution': {'Frequency': f0, 'NumModes': 10, 'Verbose': False}
    }
    solver = CmaSolverDefinitive(config)
    return solver.run()

# --- Step 1: Define Sweep Parameters ---
print('--- Unification & Q-Factor Analysis (v18.0 Definitive Edition) ---')
f0 = 300e6
lambda0 = 299792458.0 / f0
a_const = 0.001 * lambda0
N_const = 51
feed_gap = 0.01 * lambda0
L_over_lambda_sweep = np.linspace(0.1, 1.5, 31)
Ls = L_over_lambda_sweep * lambda0

# --- Step 2: Run Parallel Sweep ---
print('Starting parallel analysis for each dipole length...')
results = Parallel(n_jobs=-1)(
    delayed(run_analysis_for_length)(L, f0, a_const, N_const, feed_gap)
    for L in tqdm(Ls, desc='Overall Progress')
)
print('Main analysis sweep complete.')

# --- Step 3: Post-Process and Analyze Results ---
print('Performing final DoF and Q-Factor analysis...')
# NOTE: The |lambda_n|<1 criterion is a common heuristic. A more rigorous
# analysis might use a Q-factor threshold or modal significance.
NDoF_TCM = [np.sum(np.abs(r['lambda_n']) < 1.0) for r in results]
Q_mode1, Q_mode2 = [], []
N_Yaghjian = []

for r in tqdm(results, desc="Analyzing Results"):
    # --- Q-Factor of Dominant Modes ---
    q_sorted = r['Q_n'][np.argsort(np.abs(r['lambda_n']))]
    Q_mode1.append(q_sorted[0])
    Q_mode2.append(q_sorted[1] if len(q_sorted) > 1 else np.nan)
        
    # --- DoF from Far-Field ---
    # NOTE: A full analysis requires a convergence study w.r.t. NumModes.
    V_impressed = np.zeros(r['J_n'].shape[0])
    zn, _, _ = CmaSolverDefinitive._create_geometry(r['config']['Dipole']['Length'], r['config']['Mesh']['Segments'])
    center_node_idx = np.argmin(np.abs(zn))
    if 0 < center_node_idx < len(zn) - 1:
        V_impressed[center_node_idx - 1] = 1.0

    alpha = r['J_n'].conj().T @ V_impressed
    I_total = r['J_n'] @ (alpha / (1 + 1j*r['lambda_n']))
    
    theta = np.linspace(1e-6, np.pi - 1e-6, 361)
    E_theta_complex = CmaSolverDefinitive.calculate_radiation_properties(
        {'J_n': I_total[:, np.newaxis], **r}, theta_rad=theta
    )
    
    if np.max(np.abs(E_theta_complex)) > 1e-12:
        N_fit = fit_spherical_waves(theta, E_theta_complex[0,:], 20, 0.05)
        N_Yaghjian.append(N_fit)
    else:
        N_Yaghjian.append(0)

# --- Step 4: Plotting ---
print('Generating final report plots...')
plt.style.use('default')
styles = CmaSolverDefinitive.get_plot_styles()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('CMA Unification & Q-Factor Analysis (v18.0 Definitive Edition)', fontsize=16, fontweight='bold')

ax1.plot(L_over_lambda_sweep, NDoF_TCM, '-o', color=styles['Color1'], label='NDoF from TCM (|λn| < 1)')
ax1.plot(L_over_lambda_sweep, N_Yaghjian, '--s', color=styles['Color2'], label='NDoF from Far-Field (N_fit)')
ax1.set_xlabel('Dipole Length (L/λ)', fontweight='bold'); ax1.set_ylabel('Degrees of Freedom (NDoF)')
ax1.set_title('Internal vs. External Degrees of Freedom', fontweight='bold')
ax1.legend(); ax1.grid(True)

ax2.plot(L_over_lambda_sweep, Q_mode1, '-o', color=styles['Color1'], label='Q-Factor of Mode 1')
ax2.plot(L_over_lambda_sweep, Q_mode2, '--s', color=styles['Color2'], label='Q-Factor of Mode 2')
ax2.set_xlabel('Dipole Length (L/λ)', fontweight='bold'); ax2.set_ylabel('Energy-Based Quality Factor (Q)')
ax2.set_title('Modal Q-Factor vs. Dipole Length', fontweight='bold')
ax2.set_yscale('log'); ax2.legend(); ax2.grid(True, which="both", ls="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Fig_Report_v18_Definitive.png', dpi=300)

print('Definitive report saved to Fig_Report_v18_Definitive.png. Analysis is complete.')
plt.show()
