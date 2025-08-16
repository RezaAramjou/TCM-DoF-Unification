# run_unification_v9.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v9 import CmaSolver # Use the new solver
import matplotlib.pyplot as plt
from scipy.special import lpmv
import warnings
import seaborn as sns

def fit_spherical_waves(theta, E_pattern, max_N, error_threshold):
    """
    Robustly finds the minimum number of spherical wave modes (N_min)
    required to accurately represent a given far-field pattern.
    """
    costh = np.cos(theta)
    sinth = np.sin(theta)
    E_pattern = np.asarray(E_pattern).flatten()
    
    for N in range(1, max_N + 1):
        # Construct the basis matrix A with the correct physical basis
        A = np.zeros((len(theta), N))
        for n in range(1, N + 1):
            # The basis for E-theta is proportional to P_n^1(cos(theta))
            A[:, n-1] = lpmv(1, n, costh)
        
        # Weighted least squares for proper projection onto orthonormal basis
        W = np.diag(sinth)
        Aw = W @ A
        Ew = W @ E_pattern
        
        try:
            x, _, _, _ = np.linalg.lstsq(Aw, Ew, rcond=None)
            E_reconstructed = A @ x
            nrmse = np.linalg.norm(E_pattern - E_reconstructed) / np.linalg.norm(E_pattern)
            if nrmse < error_threshold:
                return N
        except np.linalg.LinAlgError:
            warnings.warn(f'Least-squares fit failed for N={N}.', UserWarning)
            continue
    
    warnings.warn('Fit did not converge to the error threshold. Returning max_N.', UserWarning)
    return max_N

def run_single_point(L, f0, a_const, N_const):
    """Helper function for parallel execution."""
    config = {
        'Dipole': {'Length': L, 'Radius': a_const},
        'Mesh': {'Segments': N_const, 'Strategy': 'uniform'},
        'Numerics': {'BasisFunction': 'rooftop', 'Accuracy': {'Level': 'medium'}},
        'Execution': {'Frequency': f0, 'NumModes': 20, 'Verbose': False, 'StoreZMatrix': True}
    }
    solver = CmaSolver(config)
    return solver.run()

# --- Step 1: Define Sweep ---
print('--- Unification Analysis (v9.0 Solver) ---')
f0 = 300e6
lambda0 = 299792458 / f0
a_const = 0.001 * lambda0
N_const = 41
L_over_lambda_sweep = np.linspace(0.1, 1.5, 51)
Ls = L_over_lambda_sweep * lambda0

# --- Step 2: Run Parallel Sweep ---
print('Starting data collection sweep...')
results = Parallel(n_jobs=-1)(
    delayed(run_single_point)(L, f0, a_const, N_const)
    for L in tqdm(Ls, desc='Unification Sweep')
)
print('Data collection sweep complete.')

# --- Step 3: Analyze Results ---
print('Performing TCM and Far-Field Analysis...')
NDoF_TCM = [np.sum(np.abs(r['lambda_n']) < 1.0) for r in results]
N_Yaghjian = []

for r in tqdm(results, desc="Analyzing Results"):
    # Calculate total current from all excited modes
    V = np.zeros(r['J_n'].shape[0])
    node_idx = np.argmin(np.abs(r['z_nodes']))
    feed_idx = max(0, node_idx - 1)
    V[feed_idx] = 1.0
    alpha = r['J_n'].conj().T @ V
    I_total = r['J_n'] @ (alpha / (1 + 1j*r['lambda_n']))
    
    # Calculate pattern of the TOTAL current and fit modes
    theta = np.linspace(1e-6, np.pi - 1e-6, 181)
    
    # Create a temporary result dict for the total current
    total_current_result = {
        'VersionInfo': r['VersionInfo'],
        'J_n': I_total[:, np.newaxis],
        'wavenumber': r['wavenumber'],
        'z_nodes': r['z_nodes'],
        'z_center': r['z_center'],
        'dL': r['dL']
    }
    _, E_pattern_matrix, _ = CmaSolver.calculate_radiation_properties(total_current_result, theta_rad=theta)
    E_pattern_total = E_pattern_matrix[0,:]
    
    if np.max(np.abs(E_pattern_total)) > 1e-9:
        E_pattern_normalized = np.abs(E_pattern_total) / np.max(np.abs(E_pattern_total))
        N_fit = fit_spherical_waves(theta, E_pattern_normalized, 20, 0.05)
        N_Yaghjian.append(N_fit)
    else:
        N_Yaghjian.append(0)

# --- Step 4: Plotting ---
print('Generating final unification plots...')
plt.style.use('default') # Use matplotlib defaults for consistency
styles = CmaSolver.get_plot_styles()

# Plot 1: DoF Unification
plt.figure(figsize=(8, 6))
plt.plot(L_over_lambda_sweep, NDoF_TCM, '-o', color=styles['Color1'], label='NDoF from TCM (|λn| < 1)')
plt.plot(L_over_lambda_sweep, N_Yaghjian, '--s', color=styles['Color2'], label='NDoF from Far-Field (N_fit)')
plt.xlabel('Dipole Length (L/λ)', fontweight='bold', fontsize=12)
plt.ylabel('Number of Degrees of Freedom (NDoF)', fontweight='bold', fontsize=12)
plt.title('Unification of DoF: Internal Modes vs. External Field Complexity', fontweight='bold', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('Fig_Unification_DoF.png', dpi=300)

print('Plots saved. Analysis is complete.')
plt.show()
