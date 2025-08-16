# run_unification_v15.py
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
# Ensure this import points to the final scientific solver file
from cma_solver_v15_scientific import CmaSolver 
import matplotlib.pyplot as plt
from scipy.special import lpmv
import warnings

def fit_spherical_waves(theta, E_pattern_complex, max_N, error_threshold):
    """
    Robustly finds the minimum number of spherical wave modes (N_min)
    required to accurately represent a given COMPLEX far-field pattern.

    v15 Updates:
    - Performs fit on a high-density internal Gauss-Legendre grid.
    - Adds a condition number check to warn about ill-conditioned fits.
    - Acknowledges limitation to a single polarization (E_theta).
    """
    # NOTE: This fits a single E_theta polarization. A full 4-pi analysis
    # would require vector spherical harmonics for both E_theta and E_phi.
    
    n_gauss_points = max(256, len(theta) * 2) # Use a high-density grid for accuracy
    costh_nodes, weights = np.polynomial.legendre.leggauss(n_gauss_points)
    theta_gauss = np.arccos(costh_nodes)
    
    # Interpolate the input pattern onto the high-density, optimal integration grid
    E_pattern_interp = np.interp(theta_gauss, theta, E_pattern_complex)

    for N in range(1, max_N + 1):
        # Construct the basis matrix A at the Gaussian quadrature nodes
        A = np.zeros((len(theta_gauss), N), dtype=np.float64)
        for n in range(1, N + 1):
            norm_factor = np.sqrt((2 * n + 1) / (2 * n * (n + 1)))
            A[:, n - 1] = norm_factor * lpmv(1, n, costh_nodes)

        W_sqrt = np.sqrt(weights)
        Aw = A * W_sqrt[:, np.newaxis]
        
        # Check for ill-conditioning before solving
        cond_num = np.linalg.cond(Aw)
        if cond_num > 1e10:
            warnings.warn(f'Fit matrix is ill-conditioned for N={N} (cond={cond_num:.2e}). '
                          'Fit may be unreliable.', UserWarning)

        try:
            x, _, _, _ = np.linalg.lstsq(Aw, E_pattern_interp * W_sqrt, rcond=None)
            E_reconstructed = A @ x
            
            power_true = np.sum(weights * np.abs(E_pattern_interp)**2)
            power_err = np.sum(weights * np.abs(E_pattern_interp - E_reconstructed)**2)
            
            if power_true > 1e-20 and (power_err / power_true) < error_threshold:
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
        'Execution': {'Frequency': f0, 'NumModes': 20, 'Verbose': False, 'StoreZMatrix': False}
    }
    solver = CmaSolver(config)
    return solver.run()

# --- Step 1: Define Sweep ---
print('--- Unification Analysis (v15.0 Scientific Solver) ---')
dummy_solver = CmaSolver({})
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
    # Calculate total current from all excited modes
    V_impressed = np.zeros(r['J_n'].shape[0])
    node_idx = np.argmin(np.abs(r['z_nodes']))
    if 0 < node_idx < len(r['z_nodes']) - 1:
        V_impressed[node_idx - 1] = 1.0
    
    alpha = r['J_n'].conj().T @ V_impressed
    I_total = r['J_n'] @ (alpha / (1 + 1j*r['lambda_n']))
    
    theta = np.linspace(1e-6, np.pi - 1e-6, 361) # Use a fine grid for initial pattern
    
    total_current_result = {
        'VersionInfo': r['VersionInfo'], 'J_n': I_total[:, np.newaxis],
        'wavenumber': r['wavenumber'], 'z_nodes': r['z_nodes'], 'dL': r['dL']
    }
    # Get the properly scaled complex E-field pattern
    _, _, _, E_theta_complex = CmaSolver.calculate_radiation_properties(total_current_result, theta_rad=theta)
    
    if np.max(np.abs(E_theta_complex)) > 1e-12:
        N_fit = fit_spherical_waves(theta, E_theta_complex[0,:], 20, 0.05)
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
plt.title('Unification of DoF: Internal Modes vs. External Field (v15.0)', fontweight='bold', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('Fig_Unification_DoF_v15_scientific.png', dpi=300)

print('Plots saved to Fig_Unification_DoF_v15_scientific.png. Analysis is complete.')
plt.show()
