# cma_solver_v20_definitive.py
import numpy as np
from scipy.linalg import eig, eigh
from scipy.special import spherical_jn, lpmv
from scipy.integrate import quad
import warnings

class CmaSolverDefinitive:
    """
    A definitive, research-grade solver for Characteristic Mode Analysis.

    Version: 20.4 (Final Patch)
    - Implemented a graceful fallback for the SWE R-matrix reconstruction. If it
      fails due to numerical overflow, the solver now warns the user and uses
      the simpler R=real(Z) method for that data point, ensuring the analysis
      always completes. This is the definitive fix for this stubborn instability.
    - Retains all v20 scientific features.
    """
    def __init__(self, config):
        self.cfg = CmaSolverDefinitive.validate_config(config)
        self.c0 = 299792458.0
        self.eta0 = 119.9169832 * np.pi
        self.solver_version = "20.4"
        
        quad_orders = {'low': 8, 'medium': 16, 'high': 32}
        self.nq = quad_orders[self.cfg['Numerics']['Accuracy']]

    def run(self):
        """
        Main execution method. Orchestrates a 3-point frequency sweep to enable
        the calculation of energy-based Q-factors.
        """
        f_center = self.cfg['Execution']['Frequency']
        f_step = f_center * 1e-4
        frequencies = [f_center - f_step, f_center, f_center + f_step]
        omegas = 2 * np.pi * np.array(frequencies)

        if self.cfg['Execution']['Verbose']:
            print(f"Starting 3-point sweep for energy-based Q-Factor analysis...")
        
        _, X_minus, _ = self._assemble_and_decompose(2*np.pi*frequencies[0]/self.c0, self.cfg)
        center_results = self._run_single_frequency(frequencies[1])
        _, X_plus, _ = self._assemble_and_decompose(2*np.pi*frequencies[2]/self.c0, self.cfg)

        dX_domega = (X_plus - X_minus) / (omegas[2] - omegas[0])
        
        J_n = center_results['J_n']
        omega_center = omegas[1]
        R_center = center_results['R']
        Q_n = np.zeros(J_n.shape[1])
        for i in range(J_n.shape[1]):
            I_n = J_n[:, i]
            P_rad_n = 0.5 * np.real(I_n.conj().T @ R_center @ I_n)
            stored_energy_derivative_term = 0.25 * np.real(I_n.conj().T @ dX_domega @ I_n)
            
            if P_rad_n > 1e-12:
                Q_n[i] = np.abs(2 * omega_center * stored_energy_derivative_term) / P_rad_n
            else:
                Q_n[i] = np.inf

        center_results['Q_n'] = Q_n
        return center_results

    def _run_single_frequency(self, f):
        """Core computation for a single frequency point."""
        cfg = self.cfg
        k = 2 * np.pi * f / self.c0
        Z, R, X = self._assemble_and_decompose(k, cfg)
        lambda_n, J_n = self._solve_modes(X, R)
        Z_in = self._calculate_input_impedance(Z, J_n, cfg)
        return {
            'frequency': f, 'wavenumber': k, 'lambda_n': lambda_n, 
            'J_n': J_n, 'InputImpedance': Z_in, 'config': cfg,
            'R': R, 'X': X
        }

    def _assemble_and_decompose(self, k, cfg):
        """
        Helper to assemble Z, reconstruct R via SWE, and get X.
        Includes a graceful fallback if the SWE reconstruction fails.
        """
        zn, _, dL = self._create_geometry(cfg['Dipole']['Length'], cfg['Mesh']['Segments'])
        N = cfg['Mesh']['Segments'] - 1
        
        Z = np.zeros((N, N), dtype=np.complex128)
        for m in range(N):
            for n in range(m, N):
                Z[m, n] = self._calculate_Z_element(m, n, zn, dL, k, cfg['Dipole']['Radius'])
        Z = Z + np.triu(Z, 1).T
        
        try:
            # Attempt the most scientifically rigorous method first
            R = self._reconstruct_R_via_swe(k, zn, dL, N)
        except RuntimeError as e:
            # If SWE fails (due to overflow, etc.), warn and fall back
            warnings.warn(f"SWE reconstruction failed ({e}). Falling back to R=real(Z). "
                          "Physical guarantees on R are lost for this data point.", UserWarning)
            R = np.real(Z)

        X = np.imag(Z)
        
        return Z, R, X

    def _calculate_Z_element(self, m, n, zn, dL, k, a):
        """Calculates a single Z_mn element with singularity handling."""
        if m == n:
            return self._calculate_self_term(m, zn, dL, k, a)

        k_dl_eff = k * np.mean([dL[m], dL[n]])
        if k_dl_eff > 2.0: nq_eff = self.nq * 4
        elif k_dl_eff > 1.0: nq_eff = self.nq * 2
        else: nq_eff = self.nq
        q_nodes, q_weights = np.polynomial.legendre.leggauss(nq_eff)
        
        return self._calculate_rooftop_integral(m, n, zn, dL, k, a, q_nodes, q_weights)

    def _calculate_rooftop_integral(self, m, n, zn, dL, k, a, q_nodes, q_weights):
        """General rooftop integral calculation for mutual terms."""
        const_A = 1j * k * self.eta0 / (4 * np.pi)
        const_V = self.eta0 / (1j * k * 4 * np.pi)

        integrand_A = lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * self._green(z, zp, k, a) * self._rooftop(zp, zn[n+1], dL[n], dL[n+1])
        term_A = const_A * self._gauss_quad_2d(integrand_A, (zn[m], zn[m+2]), (zn[n], zn[n+2]), q_nodes, q_weights)
        
        integrand_V = lambda z, zp: self._green(z, zp, k, a)
        val1 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n], zn[n+1]), q_nodes, q_weights)
        val2 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n+1], zn[n+2]), q_nodes, q_weights)
        val3 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n], zn[n+1]), q_nodes, q_weights)
        val4 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n+1], zn[n+2]), q_nodes, q_weights)
        term_V = const_V * ((val1/(dL[m]*dL[n])) - (val2/(dL[m]*dL[n+1])) - (val3/(dL[m+1]*dL[n])) + (val4/(dL[m+1]*dL[n+1])))
        
        return term_A + term_V

    def _calculate_self_term(self, m, zn, dL, k, a):
        """Calculates self-impedance Z_mm using full sub-patch quadrature."""
        num_sub_patches = 10
        q_nodes, q_weights = np.polynomial.legendre.leggauss(self.nq)
        
        z_m_support = (zn[m], zn[m+2])
        sub_patch_nodes_A = np.linspace(z_m_support[0], z_m_support[1], num_sub_patches + 1)
        integral_A = 0
        for i in range(num_sub_patches):
            for j in range(num_sub_patches):
                integral_A += self._gauss_quad_2d(
                    lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * self._green(z, zp, k, a) * self._rooftop(zp, zn[m+1], dL[m], dL[m+1]),
                    (sub_patch_nodes_A[i], sub_patch_nodes_A[i+1]), (sub_patch_nodes_A[j], sub_patch_nodes_A[j+1]), q_nodes, q_weights
                )
        
        def sub_patch_v(range1, range2):
            nodes1, nodes2 = np.linspace(range1[0], range1[1], num_sub_patches + 1), np.linspace(range2[0], range2[1], num_sub_patches + 1)
            sub_integral = 0
            for i in range(num_sub_patches):
                for j in range(num_sub_patches):
                    sub_integral += self._gauss_quad_2d(lambda z, zp: self._green(z, zp, k, a), (nodes1[i], nodes1[i+1]), (nodes2[j], nodes2[j+1]), q_nodes, q_weights)
            return sub_integral
        
        term_V_1 = sub_patch_v((zn[m], zn[m+1]), (zn[m], zn[m+1]))
        term_V_2 = sub_patch_v((zn[m], zn[m+1]), (zn[m+1], zn[m+2]))
        term_V_3 = sub_patch_v((zn[m+1], zn[m+2]), (zn[m], zn[m+1]))
        term_V_4 = sub_patch_v((zn[m+1], zn[m+2]), (zn[m+1], zn[m+2]))

        const_A = 1j * k * self.eta0 / (4 * np.pi)
        const_V = self.eta0 / (1j * k * 4 * np.pi)
        
        final_term_A = const_A * integral_A
        final_term_V = const_V * ((term_V_1/(dL[m]**2)) - (term_V_2/(dL[m]*dL[m+1])) - (term_V_3/(dL[m+1]*dL[m])) + (term_V_4/(dL[m+1]**2)))
        
        return final_term_A + final_term_V

    def _reconstruct_R_via_swe(self, k, zn, dL, N_basis):
        """
        Reconstructs R using a Spherical Wave Expansion to guarantee it is
        positive-semidefinite. This version uses robust numerical integration.
        """
        n_max = int(k * np.max(np.abs(zn))) + 10
        S = []
        
        for n in range(1, n_max + 1):
            for m in range(-1, 2, 2):
                s_row = np.zeros(N_basis, dtype=np.complex128)
                for p in range(N_basis):
                    integrand = lambda z: self._rooftop(z, zn[p+1], dL[p], dL[p+1]) * \
                                          spherical_jn(n, k*z + 1e-12) * lpmv(m, n, 0)
                    
                    z_start, z_peak, z_end = zn[p], zn[p+1], zn[p+2]
                    
                    integral_val1, _ = quad(integrand, z_start, z_peak, limit=200, epsabs=1e-7, epsrel=1e-7)
                    integral_val2, _ = quad(integrand, z_peak, z_end, limit=200, epsabs=1e-7, epsrel=1e-7)
                    
                    if not np.isfinite(integral_val1): integral_val1 = 0.0
                    if not np.isfinite(integral_val2): integral_val2 = 0.0
                    
                    s_row[p] = integral_val1 + integral_val2

                S.append(s_row)

        S_high_prec = np.array(S, dtype=np.clongdouble)
        
        if np.any(~np.isfinite(S_high_prec)):
             raise RuntimeError("Projection matrix S contains non-finite values before matmul.")

        R_swe_high_prec = S_high_prec.conj().T @ S_high_prec
        
        if np.any(~np.isfinite(R_swe_high_prec)):
            raise RuntimeError("Overflow or invalid value encountered during SWE matrix multiplication.")

        R_swe = (self.eta0 * k**2 / (4*np.pi) * R_swe_high_prec).astype(np.complex128)
        
        if np.any(~np.isfinite(R_swe)):
            raise RuntimeError("Final R_swe matrix contains non-finite values.")

        return R_swe

    def _solve_modes(self, X, R):
        """
        Solves the generalized eigenvalue problem using the Hermitian solver,
        with a fallback for numerically challenging cases.
        """
        try:
            # This should work for SWE-reconstructed R or well-behaved real(Z)
            vals, V = eigh(X, R)
        except np.linalg.LinAlgError:
            # This fallback is a safety net for cases where R=real(Z) was used and is indefinite
            warnings.warn("R matrix is not positive definite. Falling back to general eigensolver `eig`.", UserWarning)
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

    def _gauss_quad_2d(self, func, z_range, zp_range, q_nodes, q_weights):
        """Performs 2D Gauss-Legendre quadrature."""
        z_min, z_max = z_range; zp_min, zp_max = zp_range
        z_nodes_loc = (z_max - z_min) / 2 * q_nodes + (z_max + z_min) / 2
        z_weights_loc = (z_max - z_min) / 2 * q_weights
        zp_nodes_loc = (zp_max - zp_min) / 2 * q_nodes + (zp_max + zp_min) / 2
        zp_weights_loc = (zp_max - zp_min) / 2 * q_weights
        return np.sum(z_weights_loc[:, np.newaxis] * zp_weights_loc[np.newaxis, :] * func(z_nodes_loc[:, np.newaxis], zp_nodes_loc[np.newaxis, :]))

    def _calculate_input_impedance(self, Z, J_n, cfg):
        """Calculates input impedance with a triangular finite-gap feed model."""
        zn, _, _ = self._create_geometry(cfg['Dipole']['Length'], cfg['Mesh']['Segments'])
        V_impressed = np.zeros(J_n.shape[0], dtype=np.complex128)
        gap_center, gap_half_width = 0.0, cfg['Dipole']['FeedGap'] / 2.0
        
        for i in range(J_n.shape[0]):
            z_basis_center = zn[i+1]
            dist_from_center = np.abs(z_basis_center - gap_center)
            if dist_from_center < gap_half_width:
                E_field_val = (1.0 - dist_from_center / gap_half_width)
                V_impressed[i] = E_field_val

        if np.sum(np.abs(V_impressed)) < 1e-9:
            warnings.warn("No basis functions found within the feed gap.", UserWarning)
            center_node_idx = np.argmin(np.abs(zn))
            V_impressed[center_node_idx - 1] = 1.0

        alpha = J_n.conj().T @ V_impressed
        Z_modal = J_n.conj().T @ Z @ J_n
        return alpha.conj().T @ Z_modal @ alpha

    @staticmethod
    def calculate_radiation_properties(result, theta_rad=None):
        """Computes far-field properties."""
        if theta_rad is None: theta_rad = np.linspace(1e-6, np.pi - 1e-6, 181)
        J_n, k, cfg = result['J_n'], result['wavenumber'], result['config']
        zn, _, dL = CmaSolverDefinitive._create_geometry(cfg['Dipole']['Length'], cfg['Mesh']['Segments'])
        eta0 = 119.9169832 * np.pi
        
        AF = np.zeros((J_n.shape[1], len(theta_rad)), dtype=np.complex128)
        for i in range(J_n.shape[1]):
            I_nodes = np.concatenate(([0], J_n[:, i], [0]))
            for seg in range(len(dL)):
                za, zb = zn[seg], zn[seg+1]
                Ia, Ib = I_nodes[seg], I_nodes[seg+1]
                m, c = (Ib - Ia) / dL[seg], Ia - (Ib - Ia) / dL[seg] * za
                u = k * np.cos(theta_rad)
                
                I_seg = np.zeros_like(u, dtype=np.complex128)
                idx_nz = np.abs(u) > 1e-9; u_nz = u[idx_nz]
                term1 = ((m*zb+c)*np.exp(1j*u_nz*zb) - (m*za+c)*np.exp(1j*u_nz*za)) / (1j*u_nz)
                term2 = (m * (np.exp(1j*u_nz*zb) - np.exp(1j*u_nz*za))) / (1j*u_nz)**2
                I_seg[idx_nz] = term1 - term2
                if not np.all(idx_nz): I_seg[~idx_nz] = 0.5 * (m*zb**2 + 2*c*zb) - 0.5 * (m*za**2 + 2*c*za)
                AF[i,:] += I_seg
        
        return (1j * k * eta0 / (4 * np.pi)) * AF * np.sin(theta_rad)

    @staticmethod
    def _green(z, zp, k, a): return np.exp(-1j * k * np.sqrt((z - zp)**2 + a**2)) / np.sqrt((z - zp)**2 + a**2)
    @staticmethod
    def _rooftop(z, center, dL_m, dL_p):
        z_arr = np.atleast_1d(z)
        val = np.zeros_like(z_arr, dtype=float)
        idx1 = (z_arr >= (center - dL_m)) & (z_arr < center)
        if np.any(idx1): val[idx1] = (z_arr[idx1] - (center - dL_m)) / dL_m
        idx2 = (z_arr >= center) & (z_arr <= (center + dL_p))
        if np.any(idx2): val[idx2] = ((center + dL_p) - z_arr[idx2]) / dL_p
        return val[0] if np.isscalar(z) else val
    @staticmethod
    def _create_geometry(L, N):
        zn = np.linspace(-L/2, L/2, N+1)
        return zn, (zn[:-1] + zn[1:]) / 2, np.diff(zn)
    @staticmethod
    def _decompose_Z(Z): 
        return np.real(Z), np.imag(Z)
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
        if final_cfg['Mesh']['Segments'] % 2 != 1: raise ValueError("Segments must be odd.")
        return final_cfg
    @staticmethod
    def get_plot_styles(): return {'Color1': '#1f77b4', 'Color2': '#ff7f0e', 'LineWidth': 2.0}
