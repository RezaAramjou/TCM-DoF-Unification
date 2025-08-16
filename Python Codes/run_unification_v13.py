# run_unification_v13.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
# Ensure this import points to the new, robust, final solver file
from cma_solver_v13_robust_final import CmaSolver 
import matplotlib.pyplot as plt
from scipy.special import lpmv
import warnings

def fit_spherical_waves(theta, E_pattern_complex, max_N, error_threshold):
    """
    Robustly finds the minimum number of spherical wave modes (N_min)
    required to accurately represent a given COMPLEX far-field pattern.
    """
    n_gauss_points = len(theta) * 2 # Oversample to be safe
    costh_nodes, weights = np.polynomial.legendre.leggauss(n_gauss_points)
    theta_gauss = np.arccos(costh_nodes)
    
    E_pattern_interp_real = np.interp(theta_gauss, theta, np.real(E_pattern_complex))
    E_pattern_interp_imag = np.interp(theta_gauss, theta, np.imag(E_pattern_complex))
    E_pattern_interp = E_pattern_interp_real + 1j * E_pattern_interp_imag

    for N in range(1, max_N + 1):
        A = np.zeros((len(theta_gauss), N), dtype=np.float64)
        for n in range(1, N + 1):
            norm_factor = np.sqrt((2 * n + 1) / (2 * n * (n + 1)))
            A[:, n - 1] = norm_factor * lpmv(1, n, costh_nodes)

        W_sqrt = np.sqrt(weights)
        Aw = A * W_sqrt[:, np.newaxis]
        Ew = E_pattern_interp * W_sqrt
        
        try:
            x, _, _, _ = np.linalg.lstsq(Aw, Ew, rcond=None)
            E_reconstructed = A @ x
            
            power_true = np.sum(weights * np.abs(E_pattern_interp)**2)
            power_err = np.sum(weights * np.abs(E_pattern_interp - E_reconstructed)**2)
            
            if power_true > 1e-12 and (power_err / power_true) < error_threshold:
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
print('--- Unification Analysis (v13.3 Solver) ---')
dummy_solver = CmaSolver({'Execution': {'Frequency': 300e6}})
f0 = 300e6
lambda0 = dummy_solver.c0 / f0
a_const = 0.001 * lambda0
N_const = 51 # Must be odd
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
    V_excitation = np.zeros(r['J_n'].shape[0])
    node_idx = np.argmin(np.abs(r['z_nodes']))
    if 0 < node_idx < len(r['z_nodes']) -1:
        feed_idx = node_idx - 1
        V_excitation[feed_idx] = 1.0
    
    alpha = r['J_n'].conj().T @ V_excitation
    I_total = r['J_n'] @ (alpha / (1 + 1j*r['lambda_n']))
    
    theta = np.linspace(1e-6, np.pi - 1e-6, 181)
    
    total_current_result = {
        'VersionInfo': r['VersionInfo'],
        'J_n': I_total[:, np.newaxis],
        'wavenumber': r['wavenumber'],
        'z_nodes': r['z_nodes'],
        'z_center': r['z_center'],
        'dL': r['dL']
    }
    _, _, _, AF_matrix = CmaSolver.calculate_radiation_properties(total_current_result, theta_rad=theta)
    
    E_pattern_complex = AF_matrix[0,:] * np.sin(theta)
    
    if np.max(np.abs(E_pattern_complex)) > 1e-9:
        N_fit = fit_spherical_waves(theta, E_pattern_complex, 20, 0.05)
        N_Yaghjian.append(N_fit)
    else:
        N_Yaghjian.append(0)

# --- Step 4: Plotting ---
print('Generating final unification plots...')
plt.style.use('default')
styles = CmaSolver.get_plot_styles()

plt.figure(figsize=(8, 6))
plt.plot(L_over_lambda_sweep, NDoF_TCM, '-o', color=styles['Color1'], label='NDoF from TCM (|λn| < 1)')
plt.plot(L_over_lambda_sweep, N_Yaghjian, '--s', color=styles['Color2'], label='NDoF from Far-Field (N_fit)')
plt.xlabel('Dipole Length (L/λ)', fontweight='bold', fontsize=12)
plt.ylabel('Number of Degrees of Freedom (NDoF)', fontweight='bold', fontsize=12)
plt.title('Unification of DoF: Internal Modes vs. External Field Complexity (v13.3)', fontweight='bold', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('Fig_Unification_DoF_v13_robust_final.png', dpi=300)

print('Plots saved to Fig_Unification_DoF_v13_robust_final.png. Analysis is complete.')
plt.show()
