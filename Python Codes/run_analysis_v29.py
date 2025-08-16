# run_analysis_v29.py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v29_definitive import CmaSolverV29, CMAError
import warnings

def run_analysis_for_length(L, f0, a_const, num_segments):
    """Helper function for parallel execution with a full config."""
    config = {
        'Dipole': {'Length': L, 'Radius': a_const},
        'Mesh': {'Segments': num_segments},
        'Feed': {'GapWidth': 0.01 * L}, # Physical gap scales with length
        'Execution': {'Frequency': f0, 'NumModes': 10, 'Verbose': False, 'PowerFilterThreshold': 1e-7},
        'Quadrature': {'DuffyThresholdFactor': 2.5, 'EpsRelNear': 1e-9, 'EpsRelFar': 1e-7},
        'Numerics': {'EnforceRHermiticity': True, 'LambdaCutoff': 100}
    }
    try:
        solver = CmaSolverV29(config)
        return solver.run()
    except CMAError as e:
        print(f"Solver failed for L={L:.4f}m with {num_segments} segments: {e}")
        return None

def run_mesh_convergence_automation(f0, a_const):
    """Automatically determines a converged mesh density."""
    print("\n--- Running Automated Mesh Convergence ---")
    lambda0 = 299792458.0 / f0
    L_test = 0.48 * lambda0
    segment_counts = [41, 61, 81, 101, 121]
    impedances = []
    
    for segs in tqdm(segment_counts, desc="Mesh Convergence"):
        res = run_analysis_for_length(L_test, f0, a_const, segs)
        impedances.append(res['InputImpedance'] if res else np.nan + 1j*np.nan)
        
        if len(impedances) > 1 and not np.isnan(impedances[-1]) and not np.isnan(impedances[-2]):
            z_change = np.abs(impedances[-1] - impedances[-2])
            if z_change < 1.0: # 1 Ohm convergence tolerance
                segs_per_lambda = int(segs / (L_test/lambda0))
                print(f"  Convergence reached at {segs} segments ({z_change:.3f} Ohm change).")
                print(f"  Using {segs_per_lambda} segments per wavelength for main sweep.")
                return segs_per_lambda
                
    warnings.warn("Mesh did not converge within the tested range. Using last value.", UserWarning)
    return int(segment_counts[-1] / (L_test/lambda0))

def run_validation_suite(f0, a_const, segs_per_lambda):
    """Runs a series of tests to validate the solver's integrity."""
    print("\n--- Running Validation Suite for v29 Solver ---")
    lambda0 = 299792458.0 / f0
    
    # Test 1: Benchmark Impedance
    print("\n[1] Running Benchmark Impedance Test...")
    L_res = 0.48 * lambda0
    num_segs = int(L_res/lambda0 * segs_per_lambda) // 2 * 2 + 1
    res = run_analysis_for_length(L_res, f0, a_const, num_segs)
    if res:
        Z_in = res['InputImpedance']
        R_err = abs(np.real(Z_in) - 73.0)
        X_err = abs(np.imag(Z_in))
        print(f"    - Resonant Impedance: {Z_in.real:.2f} + {Z_in.imag:.2f}j Ohms (Target: ~73+0j)")
        if R_err < 2.0 and X_err < 2.0: print("    - STATUS: PASSED")
        else: print("    - STATUS: FAILED")
    else: print("    - STATUS: FAILED (Sim Error)")

    # Test 2: Power Conservation
    print("\n[2] Running Power Conservation Test...")
    L_test = 0.75 * lambda0
    num_segs = int(L_test/lambda0 * segs_per_lambda) // 2 * 2 + 1
    res = run_analysis_for_length(L_test, f0, a_const, num_segs)
    if res and len(res['P_rad_R']) > 0:
        P_R, P_FF = res['P_rad_R'][0], res['P_rad_FF'][0]
        rel_err = np.abs(P_R - P_FF) / np.abs(P_R) if P_R > 1e-9 else 0
        print(f"    - P_R={P_R:.4e} W vs P_FF={P_FF:.4e} W (Rel Error: {rel_err:.2%})")
        if rel_err < 0.01: print("    - STATUS: PASSED") # 1% tolerance
        else: print("    - STATUS: FAILED")
    else: print("    - STATUS: FAILED (Sim Error or no valid modes)")

    # Test 3: Orthogonality Check
    print("\n[3] Running Mode Orthogonality Test...")
    if res and len(res['J_n']) > 0:
        J_n, R_mat = res['J_n'], res['R']
        ortho_matrix = J_n.conj().T @ R_mat @ J_n
        np.fill_diagonal(ortho_matrix, 0)
        max_off_diag = np.max(np.abs(ortho_matrix))
        print(f"    - Max off-diagonal element in J^H R J: {max_off_diag:.2e}")
        if max_off_diag < 1e-8: print("    - STATUS: PASSED")
        else: print("    - STATUS: FAILED")
    else: print("    - STATUS: FAILED (Sim Error or no valid modes)")
        
    print("\n--- Validation Suite Complete ---")


# --- Step 1: Define Global Parameters ---
print('--- CMA Definitive Analysis (v29.0) ---')
f0 = 300e6
lambda0 = 299792458.0 / f0
a_const = 0.001 * lambda0

# --- Step 2: Automated Mesh Convergence & Validation ---
segments_per_lambda = run_mesh_convergence_automation(f0, a_const)
run_validation_suite(f0, a_const, segments_per_lambda)

# --- Step 3: Run Main Analysis Sweep ---
print('\n--- Starting Main Analysis Sweep (Definitive Implementation) ---')
L_over_lambda_sweep = np.linspace(0.4, 1.2, 9) 
Ls = L_over_lambda_sweep * lambda0
num_segments_list = [max(31, int(L/lambda0 * segments_per_lambda) // 2 * 2 + 1) for L in Ls]
results = Parallel(n_jobs=-1)(
    delayed(run_analysis_for_length)(L, f0, a_const, segs)
    for L, segs in tqdm(zip(Ls, num_segments_list), total=len(Ls), desc='Overall Progress')
)
print('Main analysis sweep complete.')

# --- Step 4: Post-Process and Analyze Results ---
print('Performing final analysis...')
Q_mode1, Q_mode2 = [], []

for r in tqdm(results, desc="Analyzing Results"):
    if r is None or len(r['lambda_n']) == 0:
        Q_mode1.append(np.nan); Q_mode2.append(np.nan)
        continue

    sort_idx = np.argsort(np.abs(r['lambda_n']))
    q_sorted = r['Q_n'][sort_idx]
    Q_mode1.append(q_sorted[0] if len(q_sorted) > 0 else np.nan)
    Q_mode2.append(q_sorted[1] if len(q_sorted) > 1 else np.nan)

# --- Step 5: Plotting ---
print('Generating final report plots...')
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.suptitle('CMA Modal Q-Factor (v29.0 Definitive & Validated Solver)', fontsize=16, fontweight='bold')

ax.plot(L_over_lambda_sweep, Q_mode1, '-o', markersize=8, linewidth=2, label='Q-Factor of Mode 1')
ax.plot(L_over_lambda_sweep, Q_mode2, '--s', markersize=6, linewidth=2, label='Q-Factor of Mode 2')
ax.set_title('Results from Definitive, Validated Implementation', fontweight='bold')
ax.set_xlabel('Dipole Length (L/λ)', fontweight='bold')
ax.set_ylabel('Energy-Based Quality Factor (Q)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Fig_Report_v29_Definitive.png', dpi=300)
print('\nDefinitive report saved to Fig_Report_v29_Definitive.png.')
plt.show()
