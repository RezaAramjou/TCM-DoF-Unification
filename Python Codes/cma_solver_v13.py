# cma_solver_v13.py
import numpy as np
from scipy.special import lpmv
from scipy.linalg import eig, eigh, cholesky
from scipy.integrate import quad
import warnings

class CmaSolver:
    """
    A scientifically robust, class-based solver for Characteristic Mode Analysis.

    Version: 13.2 (Final, Stabilized)
    - Corrected Z-matrix decomposition to properly enforce Hermitian structure.
    - Added numerical regularization (diagonal loading) to the R matrix to
      prevent LinAlgError from minor numerical inaccuracies, ensuring the
      Cholesky decomposition succeeds.
    - This version directly addresses the "R is not positive definite" error.
    """
    def __init__(self, config):
        self.cfg = CmaSolver.validate_config(config)
        self.c0 = 299792458.0
        self.eta0 = 119.9169832 * np.pi
        self.solver_version = "13.2"
        
        quad_orders = {'low': 4, 'medium': 8, 'high': 16}
        self.nq = quad_orders[self.cfg['Numerics']['Accuracy']['Level']]
        self.q_nodes, self.q_weights = np.polynomial.legendre.leggauss(self.nq)

    def run(self):
        """Main execution method. Runs a simulation for each frequency."""
        frequencies = np.atleast_1d(self.cfg['Execution']['Frequency'])
        results = [self._run_single_frequency(f) for f in frequencies]
        return results[0] if len(results) == 1 else results

    def _run_single_frequency(self, f):
        """Core computation for a single frequency point."""
        cfg = self.cfg
        lambda_ = self.c0 / f
        k = 2 * np.pi / lambda_
        self._validate_physical(lambda_)

        if cfg['Execution']['Verbose']:
            print(f"\n[CMA Solver] f={f/1e6:.2f}MHz, L={cfg['Dipole']['Length']:.3f}m "
                  f"({cfg['Dipole']['Length']/lambda_:.2f}λ), a={cfg['Dipole']['Radius']*1e3:.3f}mm, "
                  f"N={cfg['Mesh']['Segments']}, Basis={cfg['Numerics']['BasisFunction']}")

        zn, zc, dL = self._create_geometry(cfg['Dipole']['Length'], cfg['Mesh']['Segments'], cfg['Mesh']['Strategy'])
        
        if cfg['Execution']['Verbose']: print('Building impedance matrix... ', end='')
        Z = self._assemble_impedance(k, zn, zc, dL, cfg['Dipole']['Radius'])
        if cfg['Execution']['Verbose']: print('Done.')

        R, X = self._decompose_Z(Z)
        lambda_n, J_n = self._solve_modes(X, R)
        
        Z_in = self._calculate_input_impedance(J_n, zn, zc, lambda_n)

        result = {
            'frequency': f, 'wavelength': lambda_, 'wavenumber': k,
            'Dipole': cfg['Dipole'], 'Mesh': cfg['Mesh'],
            'z_nodes': zn, 'z_center': zc, 'dL': dL,
            'lambda_n': lambda_n, 'J_n': J_n,
            'ModalSignificance': 1 / np.abs(1 + 1j * lambda_n),
            'InputImpedance': Z_in,
            'VersionInfo': {'CmaSolver': self.solver_version, 'BasisFunction': cfg['Numerics']['BasisFunction']}
        }
        
        if cfg['Execution']['StoreZMatrix']:
            result.update({'Z_matrix': Z, 'R_matrix': R, 'X_matrix': X})
        
        if cfg['Execution']['Verbose']:
            P_rad_n, _, D_n, _ = self.calculate_radiation_properties(result)
            result.update({'P_rad_n': P_rad_n, 'Directivity_n': D_n})
            self._log_summary_table(result)
            
        return result

    def _assemble_impedance(self, k, zn, zc, dL, a):
        """Computes the full impedance matrix, exploiting symmetry."""
        basis = self.cfg['Numerics']['BasisFunction']
        N = len(zc) if basis == "pulse" else len(zn) - 2
        Z = np.zeros((N, N), dtype=np.complex128)

        for m in range(N):
            for n in range(m, N):
                Z[m, n] = self._calculate_Z_element(m, n, zn, zc, dL, k, a)
        
        Z = Z + np.triu(Z, 1).T
        return Z

    def _calculate_Z_element(self, m, n, zn, zc, dL, k, a):
        """Calculates a single Z_mn element using singularity subtraction."""
        basis = self.cfg['Numerics']['BasisFunction']
        acc = self.cfg['Numerics']['Accuracy']
        abs_tol, rel_tol = acc['AbsTol'], acc['RelTol']
        
        if basis == "pulse":
            zm = zc[m]
            zn_start, zn_end = zn[n], zn[n+1]
            if m == n:
                smooth_integrand = lambda zp: self._green(zm, zp, k, a) * k**2 - (1 / np.sqrt((zm - zp)**2 + a**2))
                singular_analytic = np.log((dL[n]/2 + np.sqrt((dL[n]/2)**2 + a**2)) / (-dL[n]/2 + np.sqrt((-dL[n]/2)**2 + a**2)))
                real_part, _ = quad(lambda zp: np.real(smooth_integrand(zp)), -dL[n]/2, dL[n]/2, epsabs=abs_tol, epsrel=rel_tol)
                imag_part, _ = quad(lambda zp: np.imag(smooth_integrand(zp)), -dL[n]/2, dL[n]/2, epsabs=abs_tol, epsrel=rel_tol)
                term_V = (real_part + singular_analytic) + 1j * imag_part
            else:
                 integrand = lambda zp: self._green(zm, zp, k, a) * k**2
                 real_part, _ = quad(lambda zp: np.real(integrand(zp)), zn_start, zn_end, epsabs=abs_tol, epsrel=rel_tol)
                 imag_part, _ = quad(lambda zp: np.imag(integrand(zp)), zn_start, zn_end, epsabs=abs_tol, epsrel=rel_tol)
                 term_V = real_part + 1j * imag_part
            d_kernel = lambda zp: self._green_d(zm, zp, k, a)
            term_A = d_kernel(zn_end) - d_kernel(zn_start)
            return (1j * self.eta0 / (4 * np.pi * k)) * (term_V + term_A)

        elif basis == "rooftop":
            const_A = 1j * k * self.eta0 / (4 * np.pi)
            const_V = self.eta0 / (1j * k * 4 * np.pi)
            integrand_A = lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * \
                                        self._green(z, zp, k, a) * \
                                        self._rooftop(zp, zn[n+1], dL[n], dL[n+1])
            term_A = const_A * self._gauss_quad_2d(integrand_A, (zn[m], zn[m+2]), (zn[n], zn[n+2]))
            integrand_V = lambda z, zp: self._green(z, zp, k, a)
            val1 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n], zn[n+1]))
            val2 = self._gauss_quad_2d(integrand_V, (zn[m], zn[m+1]), (zn[n+1], zn[n+2]))
            val3 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n], zn[n+1]))
            val4 = self._gauss_quad_2d(integrand_V, (zn[m+1], zn[m+2]), (zn[n+1], zn[n+2]))
            term_V = const_V * ((val1/(dL[m]*dL[n])) - (val2/(dL[m]*dL[n+1])) - (val3/(dL[m+1]*dL[n])) + (val4/(dL[m+1]*dL[n+1])))
            return term_A + term_V
        return 0

    def _gauss_quad_2d(self, func, z_range, zp_range):
        """Performs fast 2D Gauss-Legendre quadrature."""
        z_min, z_max = z_range
        zp_min, zp_max = zp_range
        z_nodes_loc = (z_max - z_min) / 2 * self.q_nodes + (z_max + z_min) / 2
        z_weights_loc = (z_max - z_min) / 2 * self.q_weights
        zp_nodes_loc = (zp_max - zp_min) / 2 * self.q_nodes + (zp_max + zp_min) / 2
        zp_weights_loc = (zp_max - zp_min) / 2 * self.q_weights
        vals = func(z_nodes_loc[:, np.newaxis], zp_nodes_loc[np.newaxis, :])
        real_part = np.sum(z_weights_loc[:, np.newaxis] * zp_weights_loc[np.newaxis, :] * np.real(vals))
        imag_part = np.sum(z_weights_loc[:, np.newaxis] * zp_weights_loc[np.newaxis, :] * np.imag(vals))
        return real_part + 1j * imag_part
        
    def _solve_modes(self, X, R):
        """Robustly solves the generalized eigenvalue problem X*I = lambda*R*I."""
        # Add a small diagonal loading to R to improve conditioning and
        # prevent failures for near-zero or slightly negative diagonal elements
        # that can arise from numerical errors. This is a form of regularization.
        diag_mean = np.mean(np.abs(np.diag(R)))
        diag_loading = 1e-12 * diag_mean if diag_mean > 1e-9 else 1e-9
        R_reg = R + np.eye(R.shape[0]) * diag_loading

        try:
            # Test for positive definiteness using Cholesky factorization on the regularized matrix
            cholesky(R_reg, lower=True)
            # Solve the regularized system for stability
            vals, V = eigh(X, R_reg)
        except np.linalg.LinAlgError:
            # If even the regularized matrix fails, the problem is severe.
            raise np.linalg.LinAlgError(
                "Resistance matrix R is not positive definite, even after regularization. "
                "CMA is not physically meaningful. Check mesh, frequency, or geometry. "
                "There may be a fundamental error in the Z-matrix calculation."
            )
        
        idx = np.argsort(np.abs(vals))
        N_modes = self.cfg['Execution']['NumModes']
        vals, V = vals[idx][:N_modes], V[:, idx][:, :N_modes]
        
        # R-Orthonormalization of eigenvectors using the original R
        for i in range(V.shape[1]):
            norm_factor = np.sqrt(V[:, i].conj().T @ R @ V[:, i])
            if norm_factor > 1e-12:
                V[:, i] /= norm_factor
        return vals, V

    def _calculate_input_impedance(self, J_n, zn, zc, lambda_n):
        """Calculates input impedance using the modal expansion."""
        N = J_n.shape[0]
        V_excitation = np.zeros(N, dtype=np.complex128)
        basis = self.cfg['Numerics']['BasisFunction']
        feed_idx = -1

        if basis == "rooftop":
            node_idx = np.argmin(np.abs(zn))
            if 0 < node_idx < N + 1:
                feed_idx = node_idx - 1
                V_excitation[feed_idx] = 1.0
        else: # pulse
            feed_idx = np.argmin(np.abs(zc))
            V_excitation[feed_idx] = 1.0

        if feed_idx == -1:
            warnings.warn("Could not find a valid feed point near z=0.", UserWarning)
            return np.nan + 1j*np.nan

        alpha = J_n.conj().T @ V_excitation
        Y_in = np.sum(np.abs(alpha)**2 / (1 + 1j * lambda_n))
        
        if np.abs(Y_in) < 1e-15: return np.inf + 1j*np.inf
        return 1.0 / Y_in

    @staticmethod
    def calculate_radiation_properties(result, theta_rad=None):
        """Static method to compute far-field properties from a results dict."""
        if theta_rad is None:
            theta_rad = np.linspace(1e-6, np.pi - 1e-6, 181)
        
        basis = result['VersionInfo']['BasisFunction']
        J_n, k, zn, zc, dL = result['J_n'], result['wavenumber'], result['z_nodes'], result['z_center'], result['dL']
        eta0 = 119.9169832 * np.pi
        num_modes = J_n.shape[1]
        AF = np.zeros((num_modes, len(theta_rad)), dtype=np.complex128)

        for i in range(num_modes):
            if basis == "pulse":
                AF_mode = 0
                for seg in range(len(zc)):
                    integrand = lambda z: np.exp(1j * k * z * np.cos(theta_rad))
                    res_real = np.array([quad(lambda z: np.real(np.exp(1j*k*z*ct)), zn[seg], zn[seg+1])[0] for ct in np.cos(theta_rad)])
                    res_imag = np.array([quad(lambda z: np.imag(np.exp(1j*k*z*ct)), zn[seg], zn[seg+1])[0] for ct in np.cos(theta_rad)])
                    AF_mode += J_n[seg, i] * (res_real + 1j*res_imag)
                AF[i,:] = AF_mode
            else: # rooftop (Optimized)
                I_nodes = np.concatenate(([0], J_n[:, i], [0]))
                AF_mode = 0
                for seg in range(len(zn)-1):
                    za, zb = zn[seg], zn[seg+1]
                    Ia, Ib = I_nodes[seg], I_nodes[seg+1]
                    m = (Ib - Ia) / (zb - za)
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
                    AF_mode += I_seg
                AF[i,:] = AF_mode
        
        U = (eta0 / 2.0) * np.abs(AF)**2 / (4 * np.pi**2) * np.sin(theta_rad)**2
        U_max = np.max(U, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            P_rad = np.trapz(U * 2 * np.pi * np.sin(theta_rad), theta_rad, axis=1)
            D = (4 * np.pi * U_max) / P_rad
        D[P_rad < 1e-9] = 0
        return P_rad, U, D, AF

    @staticmethod
    def _decompose_Z(Z):
        """
        Decomposes Z into its Hermitian parts, which correspond to the
        resistance and reactance matrices for CMA. This enforces the
        necessary physical structure on R and X.
        """
        Zs = (Z + Z.conj().T) / 2
        R = np.real(Zs)
        X = np.imag(Zs)
        return R, X

    def _log_summary_table(self, r):
        if not self.cfg['Execution']['Verbose'] or 'Directivity_n' not in r: return
        print("Modal Analysis Summary:")
        header = f"{'Mode':>5} {'Eigenvalue':>12} {'MS':>12} {'Directivity':>12} {'Z_in':>15}"
        print(header + "\n" + "-"*len(header))
        Z_in_str = f"{r['InputImpedance'].real:>7.2f}{r['InputImpedance'].imag:+.2f}j"
        for i in range(min(10, len(r['lambda_n']))):
            print(f"{i+1:5d} {r['lambda_n'][i]:12.4g} {r['ModalSignificance'][i]:12.4g} "
                  f"{r['Directivity_n'][i]:12.4f} " + (Z_in_str if i==0 else " "*15))

    @staticmethod
    def _green(z, zp, k, a):
        R = np.sqrt((z - zp)**2 + a**2)
        return np.exp(-1j * k * R) / R

    @staticmethod
    def _green_d(z, zp, k, a):
        R = np.sqrt((z - zp)**2 + a**2)
        return -(1 + 1j*k*R) * ((z-zp)/R**3) * np.exp(-1j*k*R)

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
        if a/lambda_ > 1/1000:
            warnings.warn(f"Thin-wire assumption may be inaccurate: a/λ={a/lambda_:.4f} > 1/1000", UserWarning)
        if L/a < 100:
            warnings.warn(f"Thin-wire assumption may be inaccurate: L/a={L/a:.2f} < 100", UserWarning)

    @staticmethod
    def _create_geometry(L, N, strat):
        if strat == 'uniform': zn = np.linspace(-L/2, L/2, N+1)
        else: idx = np.arange(N+1); zn = -L/2 * np.cos(np.pi*idx/N)
        zc = (zn[:-1] + zn[1:]) / 2; dL = np.diff(zn)
        return zn, zc, dL

    @staticmethod
    def validate_config(user_cfg):
        d = {
            'Dipole': {'Length': 0.5, 'Radius': 0.001},
            'Mesh': {'Segments': 51, 'Strategy': 'uniform'},
            'Numerics': {'BasisFunction': 'rooftop', 'Accuracy': {'Level': 'medium'}},
            'Execution': {'Frequency': 300e6, 'NumModes': 4, 'Verbose': True, 'StoreZMatrix': True},
        }
        def merge(base, overlay):
            for k, v in overlay.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    base[k] = merge(base[k], v)
                else: base[k] = v
            return base
        final_cfg = merge(d, user_cfg if user_cfg else {})
        acc = final_cfg['Numerics']['Accuracy']
        if acc['Level'] == 'low':    acc['RelTol'], acc['AbsTol'] = 1e-2, 1e-4
        elif acc['Level'] == 'high': acc['RelTol'], acc['AbsTol'] = 1e-4, 1e-8
        else:                        acc['RelTol'], acc['AbsTol'] = 1e-3, 1e-6
        if final_cfg['Mesh']['Segments'] % 2 == 0:
            raise ValueError("config.Mesh.Segments must be an odd integer for a center-fed dipole.")
        return final_cfg
        
    @staticmethod
    def get_plot_styles():
        return {'Color1': '#1f77b4', 'Color2': '#ff7f0e', 'LineWidth': 2.0}
