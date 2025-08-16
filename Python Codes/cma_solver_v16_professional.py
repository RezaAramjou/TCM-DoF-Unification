# cma_solver_v16_professional.py
import numpy as np
from scipy.linalg import eig
from scipy.integrate import quad
import warnings

class CmaSolver:
    """
    A professional-grade, class-based solver for Characteristic Mode Analysis.

    Version: 16.1 (Patched)
    - Fixed a TypeError in the __init__ method related to accessing the
      'Accuracy' level from the configuration dictionary.
    - Core functionality extended to perform multi-frequency sweeps.
    - Calculates a true, physically meaningful modal Q-factor via the
      frequency derivative of the eigenvalues (d(lambda)/d(omega)).
    - Implemented a more realistic finite-gap voltage feed model.
    - Implemented a multi-level adaptive quadrature for the Z-matrix calculation.
    """
    def __init__(self, config):
        self.cfg = CmaSolver.validate_config(config)
        self.c0 = 299792458.0
        self.eta0 = 119.9169832 * np.pi
        self.solver_version = "16.1"
        
        quad_orders = {'low': 8, 'medium': 16, 'high': 32}
        # FIX: Corrected the dictionary key access
        self.nq = quad_orders[self.cfg['Numerics']['Accuracy']]

    def run(self):
        """
        Main execution method. Runs a simulation for a single frequency or a
        frequency sweep to enable Q-factor calculations.
        """
        frequencies = np.atleast_1d(self.cfg['Execution']['Frequency'])
        
        if len(frequencies) == 1:
            return self._run_single_frequency(frequencies[0])

        # --- Frequency Sweep for Q-Factor Calculation ---
        if self.cfg['Execution']['Verbose']:
            print(f"Starting frequency sweep for Q-Factor analysis...")
        
        results_sweep = [self._run_single_frequency(f) for f in frequencies]
        
        # Calculate Q-Factor from frequency derivative
        lambdas_vs_freq = np.array([r['lambda_n'] for r in results_sweep])
        omegas = 2 * np.pi * frequencies
        
        d_lambda_d_omega = np.gradient(lambdas_vs_freq, omegas, axis=0)
        
        center_idx = len(frequencies) // 2
        center_results = results_sweep[center_idx]
        center_lambda_n = center_results['lambda_n']
        
        Q_n = np.abs(center_lambda_n / (2 * d_lambda_d_omega[center_idx, :]))
        center_results['Q_n'] = Q_n
        
        return center_results

    def _run_single_frequency(self, f):
        """Core computation for a single frequency point."""
        cfg = self.cfg
        k = 2 * np.pi * f / self.c0

        zn, zc, dL = self._create_geometry(cfg['Dipole']['Length'], cfg['Mesh']['Segments'])
        Z = self._assemble_impedance(k, zn, dL, cfg['Dipole']['Radius'])
        R, X = self._decompose_Z(Z)
        
        lambda_n, J_n = self._solve_modes(X, R)
        Z_in = self._calculate_input_impedance(Z, J_n, zn, cfg['Dipole']['FeedGap'])

        return {
            'frequency': f, 'wavenumber': k, 'z_nodes': zn, 'dL': dL,
            'lambda_n': lambda_n, 'J_n': J_n, 'InputImpedance': Z_in
        }

    def _assemble_impedance(self, k, zn, dL, a):
        """Computes the full impedance matrix."""
        N = len(zn) - 2
        Z = np.zeros((N, N), dtype=np.complex128)
        for m in range(N):
            for n in range(m, N):
                Z[m, n] = self._calculate_Z_element(m, n, zn, dL, k, a)
        return Z + np.triu(Z, 1).T.conj()

    def _calculate_Z_element(self, m, n, zn, dL, k, a):
        """Calculates a single Z_mn element with multi-level adaptive quadrature."""
        k_dl_eff = k * np.mean([dL[m], dL[n]])
        if k_dl_eff > 2.0: nq_eff = self.nq * 4
        elif k_dl_eff > 1.0: nq_eff = self.nq * 2
        else: nq_eff = self.nq
        q_nodes, q_weights = np.polynomial.legendre.leggauss(nq_eff)

        const_A = 1j * k * self.eta0 / (4 * np.pi)
        const_V = self.eta0 / (1j * k * 4 * np.pi)

        integrand_A = lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * \
                                    self._green(z, zp, k, a) * \
                                    self._rooftop(zp, zn[n+1], dL[n], dL[n+1])
        term_A = const_A * self._gauss_quad_2d(integrand_A, (zn[m], zn[m+2]), (zn[n], zn[n+2]), q_nodes, q_weights)

        integrand_V = lambda z, zp: self._green(z, zp, k, a)
        val1 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n], zn[n+1]), q_nodes, q_weights)
        val2 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n+1], zn[n+2]), q_nodes, q_weights)
        val3 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n], zn[n+1]), q_nodes, q_weights)
        val4 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n+1], zn[n+2]), q_nodes, q_weights)
        term_V = const_V * ((val1/(dL[m]*dL[n])) - (val2/(dL[m]*dL[n+1])) - (val3/(dL[m+1]*dL[n])) + (val4/(dL[m+1]*dL[n+1])))
        
        return term_A + term_V

    def _gauss_quad_2d(self, func, z_range, zp_range, q_nodes, q_weights):
        """Performs 2D Gauss-Legendre quadrature."""
        z_min, z_max = z_range
        zp_min, zp_max = zp_range
        z_nodes_loc = (z_max - z_min) / 2 * q_nodes + (z_max + z_min) / 2
        z_weights_loc = (z_max - z_min) / 2 * q_weights
        zp_nodes_loc = (zp_max - zp_min) / 2 * q_nodes + (zp_max + zp_min) / 2
        zp_weights_loc = (zp_max - zp_min) / 2 * q_weights
        return np.sum(z_weights_loc[:, np.newaxis] * zp_weights_loc[np.newaxis, :] * func(z_nodes_loc[:, np.newaxis], zp_nodes_loc[np.newaxis, :]))
        
    def _solve_modes(self, X, R):
        """Solves the generalized eigenvalue problem using the most stable general solver."""
        vals, V = eig(X, R)
        vals = np.real(vals)
        idx = np.argsort(np.abs(vals))
        N_modes = self.cfg['Execution']['NumModes']
        vals, V = vals[idx][:N_modes], V[:, idx][:, :N_modes]
        
        for i in range(V.shape[1]):
            dot_product = V[:, i].conj().T @ R @ V[:, i]
            norm_factor = np.sqrt(np.abs(dot_product))
            if norm_factor > 1e-12: V[:, i] /= norm_factor
        return vals, V

    def _calculate_input_impedance(self, Z, J_n, zn, feed_gap):
        """
        Calculates input impedance using the full modal expansion with a finite-gap feed.
        """
        V_impressed = np.zeros(J_n.shape[0], dtype=np.complex128)
        
        center_node_idx = np.argmin(np.abs(zn))
        gap_min, gap_max = -feed_gap/2, feed_gap/2
        
        for i in range(J_n.shape[0]):
            z_center_basis = zn[i+1]
            if gap_min <= z_center_basis <= gap_max:
                V_impressed[i] = 1.0 * feed_gap
        
        if np.sum(np.abs(V_impressed)) == 0:
            warnings.warn("No basis functions found within the feed gap. Z_in will be inaccurate.", UserWarning)
            V_impressed[center_node_idx - 1] = 1.0

        alpha = J_n.conj().T @ V_impressed
        Z_modal = J_n.conj().T @ Z @ J_n
        return alpha.conj().T @ Z_modal @ alpha

    @staticmethod
    def calculate_radiation_properties(result, theta_rad=None):
        """Computes far-field properties."""
        if theta_rad is None: theta_rad = np.linspace(1e-6, np.pi - 1e-6, 181)
        J_n, k, zn, dL = result['J_n'], result['wavenumber'], result['z_nodes'], result['dL']
        eta0 = 119.9169832 * np.pi
        num_modes = J_n.shape[1]
        
        AF = np.zeros((num_modes, len(theta_rad)), dtype=np.complex128)
        for i in range(num_modes):
            I_nodes = np.concatenate(([0], J_n[:, i], [0]))
            for seg in range(len(zn)-1):
                za, zb = zn[seg], zn[seg+1]
                Ia, Ib = I_nodes[seg], I_nodes[seg+1]
                m, c = (Ib - Ia) / dL[seg], Ia - (Ib - Ia) / dL[seg] * za
                u = k * np.cos(theta_rad)
                
                I_seg = np.zeros_like(u, dtype=np.complex128)
                idx_nz = np.abs(u) > 1e-9
                u_nz = u[idx_nz]
                
                term1 = ((m*zb+c)*np.exp(1j*u_nz*zb) - (m*za+c)*np.exp(1j*u_nz*za)) / (1j*u_nz)
                term2 = (m * (np.exp(1j*u_nz*zb) - np.exp(1j*u_nz*za))) / (1j*u_nz)**2
                I_seg[idx_nz] = term1 - term2
                if not np.all(idx_nz): I_seg[~idx_nz] = 0.5 * (m*zb**2 + 2*c*zb) - 0.5 * (m*za**2 + 2*c*za)
                AF[i,:] += I_seg
        
        E_theta_complex = (1j * k * eta0 / (4 * np.pi)) * AF * np.sin(theta_rad)
        return E_theta_complex

    # --- Static Helpers and Validators ---
    @staticmethod
    def _green(z, zp, k, a): return np.exp(-1j * k * np.sqrt((z - zp)**2 + a**2)) / np.sqrt((z - zp)**2 + a**2)
    @staticmethod
    def _rooftop(z, center, dL_m, dL_p):
        z_arr, val = np.atleast_1d(z), np.zeros_like(z, dtype=float)
        idx1 = (z_arr >= (center - dL_m)) & (z_arr < center)
        val[idx1] = (z_arr[idx1] - (center - dL_m)) / dL_m
        idx2 = (z_arr >= center) & (z_arr <= (center + dL_p))
        val[idx2] = ((center + dL_p) - z_arr[idx2]) / dL_p
        return val[0] if np.isscalar(z) else val
    def _validate_physical(self, lambda_):
        a, L = self.cfg['Dipole']['Radius'], self.cfg['Dipole']['Length']
        if a/lambda_ > 0.01: warnings.warn(f"Thin-wire assumption may be inaccurate: a/λ={a/lambda_:.3f} > 0.01", UserWarning)
    @staticmethod
    def _create_geometry(L, N):
        zn = np.linspace(-L/2, L/2, N+1)
        return zn, (zn[:-1] + zn[1:]) / 2, np.diff(zn)
    @staticmethod
    def _decompose_Z(Z): return np.real(Z), np.imag(Z)
    @staticmethod
    def validate_config(user_cfg):
        d = {
            'Dipole': {'Length': 0.5, 'Radius': 0.001, 'FeedGap': 0.005},
            'Mesh': {'Segments': 51},
            'Numerics': {'Accuracy': 'medium'},
            'Execution': {'Frequency': 300e6, 'NumModes': 10, 'Verbose': False},
        }
        def merge(base, overlay):
            for k, v in overlay.items():
                if isinstance(v, dict) and k in base: base[k] = merge(base[k], v)
                else: base[k] = v
            return base
        final_cfg = merge(d, user_cfg or {})
        if final_cfg['Mesh']['Segments'] % 2 == 0: raise ValueError("Segments must be odd.")
        return final_cfg
    @staticmethod
    def get_plot_styles(): return {'Color1': '#1f77b4', 'Color2': '#ff7f0e', 'LineWidth': 2.0}
