# run_analysis_v33.py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from cma_solver_v33_definitive import CmaSolverV33, CMAError
import warnings
import json
import os

CACHE_FILE = 'mesh_cache.json'

def run_analysis_for_frequency(f, L_const, a_const, num_segments):
    """Helper function for parallel execution across a frequency sweep."""
    config = {
        'Dipole': {'Length': L_const, 'Radius': a_const},
        'Mesh': {'Segments': num_segments},
        'Feed': {'GapWidth': 0.01 * L_const},
        'Execution': {'Frequency': f, 'NumModes': 10, 'Verbose': False, 'PowerFilterThreshold': 1e-7, 'ModeTrackingThreshold': 0.9},
        'Quadrature': {'DuffyThresholdFactor': 2.5, 'EpsRelNear': 1e-9, 'EpsRelFar': 1e-7},
        'Numerics': {'EnforceRHermiticity': True, 'LambdaCutoff': 100, 'RegularizationFactor': 1e-12}
    }
    try:
        solver = CmaSolverV33(config)
        return solver.run()
    except CMAError as e:
        print(f"Solver failed for f={f/1e6:.2f}MHz: {e}")
        return None

def track_modes(results):
    """
    Tracks modes across a frequency sweep using eigenvector correlation.
    
    Args:
        results (list): A list of result dictionaries from the solver.
        
    Returns:
        A list of tracked result dictionaries.
    """
    if not results or all(r is None for r in results):
        return results

    # Start with the first valid result as the reference
    valid_results = [r for r in results if r is not None and r['J_n'].shape[1] > 0]
    if not valid_results:
        return results
        
    tracked_results = [None] * len(results)
    
    # Find the index of the first valid result to initialize tracking
    first_valid_idx = next((i for i, r in enumerate(results) if r is not None and r['J_n'].shape[1] > 0), -1)
    if first_valid_idx == -1: return results # No valid results to track

    tracked_results[first_valid_idx] = valid_results[0]
    
    for i in range(first_valid_idx + 1, len(results)):
        if results[i] is None or results[i]['J_n'].shape[1] == 0:
            continue
        
        # Find the previous valid tracked result to use as a reference
        prev_res = next((tracked_results[j] for j in range(i - 1, -1, -1) if tracked_results[j] is not None), None)
        if prev_res is None: # Should not happen after first valid result is found
             tracked_results[i] = results[i]
             continue

        J_prev = prev_res['J_n']
        J_curr = results[i]['J_n']
        
        # Correlation matrix: C_ij = |J_prev_i^H * J_curr_j|
        correlation_matrix = np.abs(J_prev.conj().T @ J_curr)
        
        num_modes_prev = J_prev.shape[1]
        num_modes_curr = J_curr.shape[1]
        
        new_order = -np.ones(num_modes_curr, dtype=int)
        
        # Find the best match for each previous mode
        used_indices_curr = set()
        for j_prev in range(num_modes_prev):
            best_match_idx_curr = -1
            max_corr = -1
            # Find the best available match in the current step
            for j_curr in range(num_modes_curr):
                if j_curr not in used_indices_curr:
                    if correlation_matrix[j_prev, j_curr] > max_corr:
                        max_corr = correlation_matrix[j_prev, j_curr]
                        best_match_idx_curr = j_curr
            
            if max_corr > results[i]['config']['Execution']['ModeTrackingThreshold'] and best_match_idx_curr != -1:
                # Assign the previous mode index to the new position
                if j_prev < len(new_order):
                    new_order[best_match_idx_curr] = j_prev
                used_indices_curr.add(best_match_idx_curr)

        # Create a final ordering, placing tracked modes first, then untracked ones
        final_order = [-1] * num_modes_curr
        unassigned_slots = list(range(num_modes_curr))
        
        for curr_idx, prev_idx in enumerate(new_order):
            if prev_idx != -1 and prev_idx in unassigned_slots:
                final_order[prev_idx] = curr_idx
                unassigned_slots.remove(prev_idx)

        # Fill in any remaining untracked modes
        remaining_curr_indices = [j for j in range(num_modes_curr) if j not in final_order]
        for slot, idx in zip(unassigned_slots, remaining_curr_indices):
            final_order[slot] = idx

        # Reorder the current results
        tracked_res_i = results[i].copy()
        tracked_res_i['lambda_n'] = tracked_res_i['lambda_n'][final_order]
        tracked_res_i['J_n'] = tracked_res_i['J_n'][:, final_order]
        tracked_res_i['Q_n'] = tracked_res_i['Q_n'][final_order]
        tracked_res_i['P_rad_R'] = tracked_res_i['P_rad_R'][final_order]
        tracked_res_i['P_rad_FF'] = tracked_res_i['P_rad_FF'][final_order]
        
        tracked_results[i] = tracked_res_i
        
    return tracked_results

if __name__ == '__main__':
    print('--- CMA Definitive Analysis (v33.0) ---')
    c0 = 299792458.0
    
    # --- Define a fixed antenna for a frequency sweep ---
    L_const = 1.0 # 1 meter dipole
    a_const = 0.005 * L_const
    num_segments = 101 # A reasonably fine mesh
    
    # --- Define the frequency sweep ---
    freq_sweep_mhz = np.linspace(50, 450, 41)
    freq_sweep_hz = freq_sweep_mhz * 1e6
    
    print(f"\n--- Starting Analysis for a {L_const}m Dipole ---")
    print(f"--- Sweeping {len(freq_sweep_hz)} frequencies from {freq_sweep_mhz[0]} to {freq_sweep_mhz[-1]} MHz ---")

    results_raw = Parallel(n_jobs=-1)(
        delayed(run_analysis_for_frequency)(f, L_const, a_const, num_segments)
        for f in tqdm(freq_sweep_hz, desc='Frequency Sweep')
    )
    
    print('Main analysis sweep complete. Now tracking modes...')
    results = track_modes(results_raw)
    print('Mode tracking complete.')

    print('Performing final analysis...')
    num_modes_to_plot = 4
    all_lambdas = np.full((len(results), num_modes_to_plot), np.nan)
    all_char_angles = np.full((len(results), num_modes_to_plot), np.nan)

    for i, r in enumerate(results):
        if r is not None:
            num_avail = len(r['lambda_n'])
            all_lambdas[i, :num_avail] = r['lambda_n'][:num_modes_to_plot]
            all_char_angles[i, :num_avail] = 180 - np.rad2deg(np.arctan(r['lambda_n'][:num_modes_to_plot]))
            
    print('Generating final report plots...')
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('CMA Analysis for 1.0m Dipole (v33 with Mode Tracking)', fontsize=16, fontweight='bold')

    for i in range(num_modes_to_plot):
        ax.plot(freq_sweep_mhz, all_char_angles[:, i], '-o', markersize=4, label=f'Mode {i+1}')
    
    ax.axhline(180, color='r', linestyle='--', linewidth=2, label='Resonance (180°)')
    ax.set_title('Tracked Characteristic Angles vs. Frequency', fontweight='bold')
    ax.set_xlabel('Frequency (MHz)', fontweight='bold')
    ax.set_ylabel('Characteristic Angle (degrees)')
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 45))
    ax.legend()
    ax.grid(True, which="both", ls="--")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('Fig_Report_v33_Definitive.png', dpi=300)
    print('\nDefinitive report saved to Fig_Report_v33_Definitive.png.')
    plt.show()
