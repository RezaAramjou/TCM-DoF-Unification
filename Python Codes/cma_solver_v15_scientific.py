# cma_solver_v15_scientific.py
import numpy as np
from scipy.special import lpmv
from scipy.linalg import eig
from scipy.integrate import quad
import warnings

class CmaSolver:
    """
    A scientifically robust, class-based solver for Characteristic Mode Analysis.

    Version: 15.0 (Scientific Upgrade)
    - Implemented correct full-coupling input impedance: Z_in = a^H * Z_modal * a.
    - Implemented adaptive quadrature for Z-matrix calculation based on k*dL.
    - Removed misleading 'ModalSignificance' metric. True Q requires frequency sweep.
    - Defaulted to the general eigensolver `eig` for maximum stability.
    - Added comments acknowledging limitations (feed model, vector harmonics).
    """
    def __init__(self, config):
        self.cfg = CmaSolver.validate_config(config)
        self.c0 = 299792458.0
        self.eta0 = 119.9169832 * np.pi
        self.solver_version = "15.0"
        
        # Default quadrature order
        quad_orders = {'low': 8, 'medium': 16, 'high': 32}
        self.nq = quad_orders[self.cfg['Numerics']['Accuracy']['Level']]
        self.q_nodes, self.q_weights = np.polynomial.legendre.leggauss(self.nq)

    def run(self):
        """Main execution method."""
        frequencies = np.atleast_1d(self.cfg['Execution']['Frequency'])
        results = [self._run_single_frequency(f) for f in frequencies]
        return results[0] if len(results) == 1 else results

    def _run_single_frequency(self, f):
        """Core computation for a single frequency point."""
        cfg = self.cfg
        lambda_ = self.c0 / f
        k = 2 * np.pi / lambda_
        self._validate_physical(lambda_)

        zn, zc, dL = self._create_geometry(cfg['Dipole']['Length'], cfg['Mesh']['Segments'], cfg['Mesh']['Strategy'])
        Z = self._assemble_impedance(k, zn, zc, dL, cfg['Dipole']['Radius'])
        R, X = self._decompose_Z(Z)
        
        lambda_n, J_n = self._solve_modes(X, R)
        Z_in = self._calculate_input_impedance(Z, J_n, zn)

        result = {
            'frequency': f, 'wavelength': lambda_, 'wavenumber': k,
            'Dipole': cfg['Dipole'], 'Mesh': cfg['Mesh'],
            'z_nodes': zn, 'z_center': zc, 'dL': dL,
            'lambda_n': lambda_n, 'J_n': J_n,
            'InputImpedance': Z_in,
            'VersionInfo': {'CmaSolver': self.solver_version, 'BasisFunction': cfg['Numerics']['BasisFunction']}
        }
        # NOTE: ModalSignificance and Q_n are removed. True Q requires a frequency
        # derivative: Q_n = |lambda_n|/2 * d(omega)/d(lambda_n), which cannot
        # be computed from a single frequency run.
        
        if cfg['Execution']['StoreZMatrix']:
            result.update({'Z_matrix': Z, 'R_matrix': R, 'X_matrix': X})
        
        return result

    def _assemble_impedance(self, k, zn, zc, dL, a):
        """Computes the full impedance matrix."""
        basis = self.cfg['Numerics']['BasisFunction']
        N = len(zc) if basis == "pulse" else len(zn) - 2
        Z = np.zeros((N, N), dtype=np.complex128)

        for m in range(N):
            for n in range(m, N):
                Z[m, n] = self._calculate_Z_element(m, n, zn, zc, dL, k, a)
        
        Z = Z + np.triu(Z, 1).T.conj() # Ensure Hermitian for physical systems
        return Z

    def _calculate_Z_element(self, m, n, zn, zc, dL, k, a):
        """
        Calculates a single Z_mn element with adaptive quadrature.
        NOTE: The analytical formula for the self-term singularity is based
        on standard thin-wire MPIE formulations. For a production solver, this
        should be rigorously verified against canonical benchmarks.
        """
        basis = self.cfg['Numerics']['BasisFunction']
        
        # Adaptive quadrature: increase order for electrically large segments
        k_dl_eff = k * np.mean([dL[m], dL[n]])
        if k_dl_eff > 1.0:
            nq_eff = self.nq * 2
        else:
            nq_eff = self.nq
        q_nodes_eff, q_weights_eff = np.polynomial.legendre.leggauss(nq_eff)

        # Rooftop basis implementation
        const_A = 1j * k * self.eta0 / (4 * np.pi)
        const_V = self.eta0 / (1j * k * 4 * np.pi)

        integrand_A = lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * \
                                    self._green(z, zp, k, a) * \
                                    self._rooftop(zp, zn[n+1], dL[n], dL[n+1])
        term_A = const_A * self._gauss_quad_2d(integrand_A, (zn[m], zn[m+2]), (zn[n], zn[n+2]), q_nodes_eff, q_weights_eff)

        integrand_V = lambda z, zp: self._green(z, zp, k, a)
        val1 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n], zn[n+1]), q_nodes_eff, q_weights_eff)
        val2 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n+1], zn[n+2]), q_nodes_eff, q_weights_eff)
        val3 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n], zn[n+1]), q_nodes_eff, q_weights_eff)
        val4 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n+1], zn[n+2]), q_nodes_eff, q_weights_eff)
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
        
        vals = func(z_nodes_loc[:, np.newaxis], zp_nodes_loc[np.newaxis, :])
        return np.sum(z_weights_loc[:, np.newaxis] * zp_weights_loc[np.newaxis, :] * vals)
        
    def _solve_modes(self, X, R):
        """
        Solves the generalized eigenvalue problem X*I = lambda*R*I using the
        most stable general solver.
        """
        # The general eigensolver `eig` is the most robust choice as it does
        # not require R to be positive-definite, handling all numerical cases.
        vals, V = eig(X, R)

        # Eigenvalues should be real for a lossless system. A small imaginary
        # part can indicate numerical noise or discretization errors.
        if np.max(np.abs(np.imag(vals))) > 1e-9 * np.max(np.abs(np.real(vals))):
            warnings.warn("Eigenvalues have non-trivial imaginary parts, indicating potential numerical issues.", UserWarning)
        
        vals = np.real(vals)
        idx = np.argsort(np.abs(vals))
        N_modes = self.cfg['Execution']['NumModes']
        vals, V = vals[idx][:N_modes], V[:, idx][:, :N_modes]
        
        # R-Orthonormalization of eigenvectors
        for i in range(V.shape[1]):
            dot_product = V[:, i].conj().T @ R @ V[:, i]
            norm_factor = np.sqrt(np.abs(dot_product))
            if norm_factor > 1e-12:
                V[:, i] /= norm_factor
        return vals, V

    def _calculate_input_impedance(self, Z, J_n, zn):
        """
        Calculates input impedance using the scientifically correct full modal
        expansion: Z_in = alpha^H * Z_modal * alpha.
        NOTE: The feed model is a simple delta-gap voltage source at z=0, which
        is an approximation.
        """
        N = J_n.shape[0]
        V_impressed = np.zeros(N, dtype=np.complex128)
        
        # Find rooftop basis function closest to the center (z=0) to excite
        node_idx = np.argmin(np.abs(zn))
        if 0 < node_idx < N + 1:
            V_impressed[node_idx - 1] = 1.0 # 1V excitation
        else:
            warnings.warn("Could not find a valid feed point near z=0.", UserWarning)
            return np.nan + 1j*np.nan

        # Project impressed voltage vector onto the modal basis to get modal excitation coefficients
        alpha = J_n.conj().T @ V_impressed
        
        # Project the full impedance matrix onto the modal basis
        Z_modal = J_n.conj().T @ Z @ J_n
        
        # Calculate input impedance using the full quadratic form
        Z_in = alpha.conj().T @ Z_modal @ alpha
        
        return Z_in

    @staticmethod
    def calculate_radiation_properties(result, theta_rad=None):
        """
        Computes far-field properties. Returns the properly scaled complex E-field.
        """
        if theta_rad is None:
            theta_rad = np.linspace(1e-6, np.pi - 1e-6, 181)
        
        J_n, k, zn, dL = result['J_n'], result['wavenumber'], result['z_nodes'], result['dL']
        eta0 = 119.9169832 * np.pi
        num_modes = J_n.shape[1]
        
        # Calculate Array Factor (AF)
        AF = np.zeros((num_modes, len(theta_rad)), dtype=np.complex128)
        for i in range(num_modes):
            I_nodes = np.concatenate(([0], J_n[:, i], [0]))
            for seg in range(len(zn)-1):
                za, zb = zn[seg], zn[seg+1]
                Ia, Ib = I_nodes[seg], I_nodes[seg+1]
                m = (Ib - Ia) / dL[seg]
                c = Ia - m * za
                u = k * np.cos(theta_rad)
                
                I_seg = np.zeros_like(u, dtype=np.complex128)
                idx_nz = np.abs(u) > 1e-9
                u_nz = u[idx_nz]
                
                term1 = ((m*zb+c)*np.exp(1j*u_nz*zb) - (m*za+c)*np.exp(1j*u_nz*za)) / (1j*u_nz)
                term2 = (m * (np.exp(1j*u_nz*zb) - np.exp(1j*u_nz*za))) / (1j*u_nz)**2
                I_seg[idx_nz] = term1 - term2
                if not np.all(idx_nz):
                    I_seg[~idx_nz] = 0.5 * (m*zb**2 + 2*c*zb) - 0.5 * (m*za**2 + 2*c*za)
                AF[i,:] += I_seg
        
        # Calculate the complex E-field (ignoring 1/r dependence)
        E_theta_complex = (1j * k * eta0 / (4 * np.pi)) * AF * np.sin(theta_rad)
        
        # Calculate radiation intensity U and directivity D
        U = (eta0 / 2) * np.abs(AF)**2 / (4 * np.pi**2) * np.sin(theta_rad)**2
        P_rad = np.trapz(U * 2 * np.pi * np.sin(theta_rad), theta_rad, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            D = (4 * np.pi * np.max(U, axis=1)) / P_rad
        D[P_rad < 1e-9] = 0
        
        return P_rad, U, D, E_theta_complex

    # --- Static Helpers and Validators ---
    @staticmethod
    def _green(z, zp, k, a):
        R = np.sqrt((z - zp)**2 + a**2)
        return np.exp(-1j * k * R) / R

    @staticmethod
    def _rooftop(z, center, dL_m, dL_p):
        z_arr = np.atleast_1d(z)
        val = np.zeros_like(z_arr, dtype=float)
        idx1 = (z_arr >= (center - dL_m)) & (z_arr < center)
        val[idx1] = (z_arr[idx1] - (center - dL_m)) / dL_m
        idx2 = (z_arr >= center) & (z_arr <= (center + dL_p))
        val[idx2] = ((center + dL_p) - z_arr[idx2]) / dL_p
        return val[0] if np.isscalar(z) else val

    def _validate_physical(self, lambda_):
        a, L = self.cfg['Dipole']['Radius'], self.cfg['Dipole']['Length']
        if a/lambda_ > 0.01: warnings.warn(f"Thin-wire assumption may be inaccurate: a/λ={a/lambda_:.3f} > 0.01", UserWarning)
        if L/a < 50: warnings.warn(f"Thin-wire assumption may be inaccurate: L/a={L/a:.2f} < 50", UserWarning)

    @staticmethod
    def _create_geometry(L, N, strat):
        if strat == 'uniform': zn = np.linspace(-L/2, L/2, N+1)
        else: idx = np.arange(N+1); zn = -L/2 * np.cos(np.pi*idx/N)
        zc = (zn[:-1] + zn[1:]) / 2; dL = np.diff(zn)
        return zn, zc, dL

    @staticmethod
    def _decompose_Z(Z):
        R = np.real(Z)
        X = np.imag(Z)
        return R, X

    @staticmethod
    def validate_config(user_cfg):
        d = {
            'Dipole': {'Length': 0.5, 'Radius': 0.001},
            'Mesh': {'Segments': 51, 'Strategy': 'uniform'},
            'Numerics': {'BasisFunction': 'rooftop', 'Accuracy': {'Level': 'medium'}},
            'Execution': {'Frequency': 300e6, 'NumModes': 10, 'Verbose': False, 'StoreZMatrix': True},
        }
        def merge(base, overlay):
            for k, v in overlay.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    base[k] = merge(base[k], v)
                else: base[k] = v
            return base
        final_cfg = merge(d, user_cfg if user_cfg else {})
        if final_cfg['Mesh']['Segments'] % 2 == 0:
            raise ValueError("config.Mesh.Segments must be an odd integer for a center-fed dipole.")
        return final_cfg
        
    @staticmethod
    def get_plot_styles():
        return {'Color1': '#1f77b4', 'Color2': '#ff7f0e', 'LineWidth': 2.0}
