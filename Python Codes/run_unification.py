# run_unification_final.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_final import CmaSolver
import matplotlib.pyplot as plt
from scipy.special import lpmv # <-- ADDED THIS LINE

def fit_spherical_waves(theta, E_pattern, max_N, error_threshold):
    """
    Robustly finds the minimum number of spherical wave modes (N_min)
    required to accurately represent a given far-field pattern.
    """
    costh = np.cos(theta)
    E_pattern = np.asarray(E_pattern).flatten()
    for N in range(1, max_N + 1):
        A = np.zeros((len(theta), N))
        for n in range(1, N + 1):
            A[:, n-1] = lpmv(1, n, costh) # P_n^1(cos(theta))

        if np.linalg.cond(A) > 1e10:
            warnings.warn(f'Fit matrix is ill-conditioned (cond={np.linalg.cond(A):.2e}). Using pseudoinverse.', UserWarning)
            x, _, _, _ = np.linalg.lstsq(A, E_pattern, rcond=None)
        else:
            x = np.linalg.solve(A.T @ A, A.T @ E_pattern)
        
        E_reconstructed = A @ x
        nrmse = np.linalg.norm(E_pattern - E_reconstructed) / np.linalg.norm(E_pattern)
        
        if nrmse < error_threshold:
            return N
    
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
print('--- Unification Analysis (v7.0 Solver) ---')
f0 = 300e6
lambda0 = 299792458 / f0
a_const = 0.001 * lambda0
N_const = 41
L_over_lambda_sweep = np.linspace(0.1, 1.5, 51)
Ls = L_over_lambda_sweep * lambda0

# --- Step 2: Run Parallel Sweep ---
print('Starting data collection sweep...')
# Use joblib for parallel execution with a progress bar
results = Parallel(n_jobs=-1)(
    delayed(run_single_point)(L, f0, a_const, N_const)
    for L in tqdm(Ls, desc='Unification Sweep')
)
print('Data collection sweep complete.')

# --- Step 3: Analyze Results ---
print('Performing TCM and Far-Field Analysis...')
NDoF_TCM = [np.sum(np.abs(r['lambda_n']) < 1.0) for r in results]
N_Yaghjian = []
effective_radius_a = []
physical_half_length = [r['Dipole']['Length'] / 2 for r in results]

for r in tqdm(results, desc="Analyzing Results"):
    # Calculate total current
    V = np.zeros(r['J_n'].shape[0])
    node_idx = np.argmin(np.abs(r['z_nodes']))
    feed_idx = max(0, node_idx - 1)
    V[feed_idx] = 1.0
    I_total = np.linalg.solve(r['Z_matrix'], V)
    
    # Calculate pattern and fit modes
    theta = np.linspace(0, np.pi, 181)
    _, E_pattern_matrix, _ = CmaSolver.calculate_radiation_properties(r)
    E_pattern = E_pattern_matrix[0,:] / np.max(E_pattern_matrix[0,:])
    
    N_fit = fit_spherical_waves(theta, E_pattern, 20, 0.05)
    N_Yaghjian.append(N_fit)
    effective_radius_a.append((N_fit + 0.5) / r['wavenumber'])

# --- Step 4: Plotting ---
print('Generating final unification plots...')
styles = CmaSolver.get_plot_styles()
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: DoF Unification
plt.figure(figsize=(8, 6))
plt.plot(L_over_lambda_sweep, NDoF_TCM, '-o', color=styles['Color1'], label='NDoF from TCM (|λn| < 1)')
plt.plot(L_over_lambda_sweep, N_Yaghjian, '--s', color=styles['Color2'], label='NDoF from Far-Field (NYaghjian)')
plt.xlabel('Dipole Length (L/λ)', fontweight='bold')
plt.ylabel('Number of Degrees of Freedom (NDoF)', fontweight='bold')
plt.title('Unification of DoF: Internal Modes vs. External Field Complexity', fontweight='bold')
plt.legend(loc='upper left')
plt.savefig('Fig_Unification_DoF.png', dpi=300)

# Plot 2: Effective Size
plt.figure(figsize=(8, 6))
plt.plot(L_over_lambda_sweep, effective_radius_a, '-d', color='green', label='Effective Reactive Radius a_eff')
plt.plot(L_over_lambda_sweep, physical_half_length, '--^', color='purple', label='Physical Half-Length L/2')
plt.xlabel('Dipole Length (L/λ)', fontweight='bold')
plt.ylabel('Radius (meters)', fontweight='bold')
plt.title('Effective Radiating Size vs. Physical Size', fontweight='bold')
plt.legend(loc='upper left')
plt.savefig('Fig_Unification_Size.png', dpi=300)

print('Plots saved. Analysis is complete.')
plt.show()
