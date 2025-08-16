# run_project_unification.py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v34_definitive import CmaSolverV34, CMAError
from scipy.special import lpmv
import warnings
import json
import os

RESULTS_CACHE_FILE = 'results_cache.dat'

# Helper functions for saving/loading numpy arrays and complex numbers in json
def save_results(filepath, data):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return {'_kind_': 'ndarray', 'value': obj.tolist()}
            # *** BUG FIX IS HERE ***
            # Added a handler for complex numbers
            if np.iscomplexobj(obj):
                return {'_kind_': 'complex', 'real': obj.real, 'imag': obj.imag}
            return json.JSONEncoder.default(self, obj)
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)

def load_results(filepath):
    def json_numpy_obj_hook(dct):
        if isinstance(dct, dict) and '_kind_' in dct:
            if dct['_kind_'] == 'ndarray':
                return np.array(dct['value'])
            # *** BUG FIX IS HERE ***
            # Added a handler to reconstruct complex numbers
            if dct['_kind_'] == 'complex':
                return dct['real'] + 1j * dct['imag']
        return dct
    with open(filepath, 'r') as f:
        return json.load(f, object_hook=json_numpy_obj_hook)


def run_analysis_for_length(L, f0, a_const, num_segments):
    """Helper function for parallel execution of a single dipole length."""
    config = {
        'Dipole': {'Length': L, 'Radius': a_const},
        'Mesh': {'Segments': num_segments},
        'Execution': {'Frequency': f0, 'NumModes': 15, 'Verbose': False}
    }
    try:
        solver = CmaSolverV34(config)
        return solver.run()
    except CMAError as e:
        print(f"Solver failed for L={L:.4f}m: {e}")
        return None

def fit_spherical_waves(theta, E_pattern_complex, max_N=20, error_threshold=0.01):
    """
    Performs a spherical wave decomposition to find the minimum number of modes N
    required to accurately represent the far-field pattern.
    """
    n_gauss = 512
    costh_nodes, weights = np.polynomial.legendre.leggauss(n_gauss)
    theta_gauss = np.arccos(costh_nodes)
    
    E_interp_real = np.interp(theta_gauss, theta, np.real(E_pattern_complex))
    E_interp_imag = np.interp(theta_gauss, theta, np.imag(E_pattern_complex))
    E_interp = E_interp_real + 1j * E_interp_imag

    for N in range(1, max_N + 1):
        A = np.zeros((n_gauss, N))
        for n in range(1, N + 1):
            norm = np.sqrt((2 * n + 1) / (2 * n * (n + 1)))
            A[:, n - 1] = norm * lpmv(1, n, costh_nodes)

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, E_interp, rcond=None)
            E_reconstructed = A @ coeffs
            
            power_true = np.sum(weights * np.abs(E_interp)**2)
            power_err = np.sum(weights * np.abs(E_interp - E_reconstructed)**2)
            
            if power_true > 1e-20 and (power_err / power_true) < error_threshold:
                return N
        except np.linalg.LinAlgError:
            warnings.warn(f'Least-squares fit failed for N={N}.', UserWarning)
            return N - 1
            
    return max_N

if __name__ == '__main__':
    print('--- Project Unification Analysis (v34.0) ---')
    f0 = 300e6
    lambda0 = 299792458.0 / f0
    a_const = 0.001 * lambda0
    
    segments_per_lambda = 80
    
    if os.path.exists(RESULTS_CACHE_FILE):
        print(f"\n--- Loading cached simulation results from {RESULTS_CACHE_FILE} ---")
        results = load_results(RESULTS_CACHE_FILE)
    else:
        print(f"\n--- Starting Analysis for a Dipole at {f0/1e6} MHz ---")
        L_over_lambda_sweep = np.linspace(0.1, 1.5, 29) 
        Ls = L_over_lambda_sweep * lambda0
        num_segments_list = [max(31, int(L/lambda0 * segments_per_lambda) // 2 * 2 + 1) for L in Ls]

        results = Parallel(n_jobs=-1)(
            delayed(run_analysis_for_length)(L, f0, a_const, segs)
            for L, segs in tqdm(zip(Ls, num_segments_list), total=len(Ls), desc='Length Sweep')
        )
        print('Main analysis sweep complete. Caching results...')
        save_results(RESULTS_CACHE_FILE, results)
        print(f"Results saved to {RESULTS_CACHE_FILE}.")

    print('Now performing unification analysis...')
    
    # --- Phase 4 Analysis ---
    L_over_lambda_sweep = np.linspace(0.1, 1.5, 29) # Ensure this matches the sweep
    NDoF_TCM = []
    NDoF_Yaghjian = []

    for r in tqdm(results, desc="Unification Analysis"):
        if r is None:
            NDoF_TCM.append(np.nan)
            NDoF_Yaghjian.append(np.nan)
            continue

        significant_modes = np.sum(np.abs(r['lambda_n']) < 1.0)
        NDoF_TCM.append(significant_modes)
        
        theta_pts = np.linspace(1e-6, np.pi - 1e-6, 361)
        E_pattern_mode1 = r['E_theta'][0, :]
        
        if np.max(np.abs(E_pattern_mode1)) > 1e-12:
            N_fit = fit_spherical_waves(theta_pts, E_pattern_mode1)
            NDoF_Yaghjian.append(N_fit)
        else:
            NDoF_Yaghjian.append(0)

    print('Generating the Grand Unification plot...')
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('The Grand Unification: DoF vs. Electrical Length', fontsize=16, fontweight='bold')

    ax.plot(L_over_lambda_sweep, NDoF_TCM, '-o', markersize=6, label='DoF from Characteristic Modes (|λn| < 1)')
    ax.plot(L_over_lambda_sweep, NDoF_Yaghjian, '--s', markersize=5, label="DoF from Far-Field Complexity (Yaghjian's N)")
    
    ax.set_title('Unification of Internal Current Modes and External Field Complexity', fontweight='bold')
    ax.set_xlabel('Dipole Length (L/λ)', fontweight='bold')
    ax.set_ylabel('Number of Degrees of Freedom (NDoF)')
    ax.legend()
    ax.grid(True, which="both", ls="--")
    ax.set_yticks(np.arange(0, max(max(np.nan_to_num(NDoF_TCM)), max(np.nan_to_num(NDoF_Yaghjian))) + 2, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('Fig_Project_Unification.png', dpi=300)
    print('\nDefinitive project report plot saved to Fig_Project_Unification.png.')
    plt.show()
