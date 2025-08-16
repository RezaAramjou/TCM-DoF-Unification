# run_unification_v16.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v16_professional import CmaSolver 
import matplotlib.pyplot as plt
from scipy.special import lpmv
import warnings

def fit_spherical_waves(theta, E_pattern_complex, max_N, error_threshold):
    """
    Robustly finds the minimum number of spherical wave modes (N_min)
    required to accurately represent a given COMPLEX far-field pattern.
    """
    n_gauss_points = max(256, len(theta) * 2)
    costh_nodes, weights = np.polynomial.legendre.leggauss(n_gauss_points)
    theta_gauss = np.arccos(costh_nodes)
    E_pattern_interp = np.interp(theta_gauss, theta, E_pattern_complex)

    for N in range(1, max_N + 1):
        A = np.zeros((len(theta_gauss), N), dtype=np.float64)
        for n in range(1, N + 1):
            A[:, n - 1] = np.sqrt((2 * n + 1) / (2 * n * (n + 1))) * lpmv(1, n, costh_nodes)

        Aw = A * np.sqrt(weights)[:, np.newaxis]
        if np.linalg.cond(Aw) > 1e10:
            warnings.warn(f'Fit matrix is ill-conditioned for N={N}.', UserWarning)

        try:
            x, _, _, _ = np.linalg.lstsq(Aw, E_pattern_interp * np.sqrt(weights), rcond=None)
            E_reconstructed = A @ x
            power_true = np.sum(weights * np.abs(E_pattern_interp)**2)
            power_err = np.sum(weights * np.abs(E_pattern_interp - E_reconstructed)**2)
            if power_true > 1e-20 and (power_err / power_true) < error_threshold:
                return N
        except np.linalg.LinAlgError:
            warnings.warn(f'Least-squares fit failed for N={N}.', UserWarning)
    
    warnings.warn('Fit did not converge. Returning max_N.', UserWarning)
    return max_N

def run_analysis_for_length(L, f0, a_const, N_const, feed_gap):
    """
    Helper function for parallel execution. Runs a frequency sweep for each L.
    """
    # Define a small frequency band around f0 for Q-factor calculation
    freq_sweep = np.linspace(f0 * 0.99, f0 * 1.01, 21) # 21 points for stable derivative
    
    config = {
        'Dipole': {'Length': L, 'Radius': a_const, 'FeedGap': feed_gap},
        'Mesh': {'Segments': N_const},
        'Numerics': {'Accuracy': 'medium'},
        'Execution': {'Frequency': freq_sweep, 'NumModes': 10, 'Verbose': False}
    }
    solver = CmaSolver(config)
    return solver.run()

# --- Step 1: Define Sweep Parameters ---
print('--- Unification & Q-Factor Analysis (v16.0 Professional Solver) ---')
f0 = 300e6
lambda0 = 299792458.0 / f0
a_const = 0.001 * lambda0
N_const = 51
feed_gap = 0.01 * lambda0 # Finite gap of 1% of a wavelength
L_over_lambda_sweep = np.linspace(0.1, 1.5, 31) # Reduced points for faster demo
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
NDoF_TCM = []
N_Yaghjian = []
Q_mode1 = []
Q_mode2 = []

for r in tqdm(results, desc="Analyzing Results"):
    # --- DoF from TCM ---
    NDoF_TCM.append(np.sum(np.abs(r['lambda_n']) < 1.0))
    
    # --- DoF from Far-Field ---
    V_impressed = np.zeros(r['J_n'].shape[0])
    center_node_idx = np.argmin(np.abs(r['z_nodes']))
    V_impressed[center_node_idx - 1] = 1.0 # Simplified excitation for pattern
    alpha = r['J_n'].conj().T @ V_impressed
    I_total = r['J_n'] @ (alpha / (1 + 1j*r['lambda_n']))
    
    theta = np.linspace(1e-6, np.pi - 1e-6, 361)
    E_theta_complex = CmaSolver.calculate_radiation_properties(
        {'J_n': I_total[:, np.newaxis], **r}, theta_rad=theta
    )
    
    if np.max(np.abs(E_theta_complex)) > 1e-12:
        N_fit = fit_spherical_waves(theta, E_theta_complex[0,:], 20, 0.05)
        N_Yaghjian.append(N_fit)
    else:
        N_Yaghjian.append(0)
        
    # --- Q-Factor of Dominant Modes ---
    # Sort Q by the eigenvalue magnitude to find Q of mode 1, 2, etc.
    q_sorted = r['Q_n'][np.argsort(np.abs(r['lambda_n']))]
    Q_mode1.append(q_sorted[0])
    if len(q_sorted) > 1:
        Q_mode2.append(q_sorted[1])
    else:
        Q_mode2.append(np.nan)

# --- Step 4: Plotting ---
print('Generating final report plots...')
plt.style.use('default')
styles = CmaSolver.get_plot_styles()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('CMA Unification & Q-Factor Analysis (v16.0 Professional)', fontsize=16, fontweight='bold')

# Plot 1: DoF Unification
ax1.plot(L_over_lambda_sweep, NDoF_TCM, '-o', color=styles['Color1'], label='NDoF from TCM (|λn| < 1)')
ax1.plot(L_over_lambda_sweep, N_Yaghjian, '--s', color=styles['Color2'], label='NDoF from Far-Field (N_fit)')
ax1.set_xlabel('Dipole Length (L/λ)', fontweight='bold')
ax1.set_ylabel('Number of Degrees of Freedom (NDoF)')
ax1.set_title('Unification of Internal vs. External Degrees of Freedom', fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True)

# Plot 2: Modal Q-Factor
ax2.plot(L_over_lambda_sweep, Q_mode1, '-o', color=styles['Color1'], label='Q-Factor of Mode 1')
ax2.plot(L_over_lambda_sweep, Q_mode2, '--s', color=styles['Color2'], label='Q-Factor of Mode 2')
ax2.set_xlabel('Dipole Length (L/λ)', fontweight='bold')
ax2.set_ylabel('Modal Quality Factor (Q)')
ax2.set_title('Modal Q-Factor vs. Dipole Length', fontweight='bold')
ax2.set_yscale('log')
ax2.legend(loc='upper right')
ax2.grid(True, which="both", ls="--")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Fig_Report_v16_Professional.png', dpi=300)

print('Professional report saved to Fig_Report_v16_Professional.png. Analysis is complete.')
plt.show()
