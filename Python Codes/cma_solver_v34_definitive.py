# cma_solver_v34_definitive.py
import numpy as np
from scipy.linalg import eigh
from scipy.special import spherical_jn, lpmv
from scipy.integrate import quad, dblquad
import warnings

class CMAError(Exception):
    """Custom exception for solver-specific errors."""
    pass

class CmaSolverV34:
    """
    A definitive, research-grade solver for Characteristic Mode Analysis.

    Version: 34.0 (Project Unification Edition)
    - BUG FIX: Corrected the default number of points in the far-field pattern
      calculation to match the analysis script's requirements (361 points).
    - This version is functionally identical to v33 but is versioned to
      accompany the project-specific analysis script.
    """
    def __init__(self, config):
        self.cfg = CmaSolverV34.validate_config(config)
        self.c0 = 299792458.0
        self.eta0 = 119.9169832 * np.pi
        self.solver_version = "34.0"

    def run(self):
        f = self.cfg['Execution']['Frequency']
        if self.cfg['Execution']['Verbose']: print(f"Analyzing at {f/1e6:.2f} MHz with v{self.solver_version}...")

        omega = 2 * np.pi * f
        k = omega / self.c0

        Z, _ = self._assemble_matrices_uniform_singular(k)
        R = self._reconstruct_R_via_swe(k)
        X = np.imag(Z)

        lambda_n, J_n, _ = self._solve_and_filter_modes(X, R)
        
        # We only need the raw modes and patterns for the project
        E_theta = self.calculate_radiation_pattern(k, J_n)

        return {
            'frequency': f, 'wavenumber': k, 'lambda_n': lambda_n,
            'J_n': J_n, 'E_theta': E_theta, 'config': self.cfg,
            'R': R, 'X': X
        }

    def _assemble_matrices_uniform_singular(self, k):
        zn, z_centers, dL = self._create_geometry(self.cfg['Dipole']['Length'], self.cfg['Mesh']['Segments'])
        N = self.cfg['Mesh']['Segments'] - 1
        a = self.cfg['Dipole']['Radius']
        
        Z = np.zeros((N, N), dtype=np.complex128)
        dZ_dk = np.zeros((N, N), dtype=np.complex128) # Kept for compatibility, but not used in project analysis

        for m in range(N):
            for n in range(m, N):
                dist_threshold = self.cfg['Quadrature']['DuffyThresholdFactor'] * min(dL[m], dL[n])
                if m == n:
                    Z[m, n], dZ_dk[m, n] = self._calculate_self_term_full_analytical(m, zn, dL, k, a)
                else:
                    z_m_support = (zn[m], zn[m+2]); z_n_support = (zn[n], zn[n+2])
                    z_kernel = lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * self._integrand_kernel(z, zp, k, a) * self._rooftop(zp, zn[n+1], dL[n], dL[n+1])
                    dzdk_kernel = lambda z, zp: self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * self._integrand_kernel_dk(z, zp, k, a) * self._rooftop(zp, zn[n+1], dL[n], dL[n+1])
                    
                    min_dist = abs(z_centers[m] - z_centers[n]) - (dL[m] + dL[n])/2
                    if min_dist < dist_threshold:
                        eps = self.cfg['Quadrature']['EpsRelNear']
                        Z[m, n] = self._duffy_quadrature(z_kernel, z_m_support, z_n_support, eps)
                        dZ_dk[m, n] = self._duffy_quadrature(dzdk_kernel, z_m_support, z_n_support, eps)
                    else:
                        eps = self.cfg['Quadrature']['EpsRelFar']
                        Z[m, n] = dblquad(z_kernel, z_n_support[0], z_n_support[1], z_m_support[0], z_m_support[1], epsrel=eps)[0]
                        dZ_dk[m, n] = dblquad(dzdk_kernel, z_n_support[0], z_n_support[1], z_m_support[0], z_m_support[1], epsrel=eps)[0]
        
        Z = Z + np.triu(Z, 1).T.conj()
        dZ_dk = dZ_dk + np.triu(dZ_dk, 1).T.conj()
        return Z, dZ_dk

    def _duffy_quadrature(self, kernel_func, z_range, zp_range, eps):
        z_min, z_max = z_range; zp_min, zp_max = zp_range
        def integrand1(u, v):
            z = z_min + u * (z_max - z_min); zp = zp_min + u * v * (zp_max - zp_min)
            jacobian = u * (z_max - z_min) * (zp_max - zp_min)
            return kernel_func(z, zp) * jacobian
        def integrand2(u, v):
            z = z_min + u * v * (z_max - z_min); zp = zp_min + v * (zp_max - zp_min)
            jacobian = v * (z_max - z_min) * (zp_max - zp_min)
            return kernel_func(z, zp) * jacobian
        res1 = dblquad(integrand1, 0, 1, 0, 1, epsrel=eps)[0]
        res2 = dblquad(integrand2, 0, 1, 0, 1, epsrel=eps)[0]
        return res1 + res2

    def _calculate_self_term_full_analytical(self, m, zn, dL, k, a):
        eps = self.cfg['Quadrature']['EpsRelNear']
        const = 1 / (4j * np.pi * k * self.eta0)
        l1, l2 = dL[m], dL[m+1]
        def I_sing(L, a_val):
            if L < 1e-12: return 0
            return (L/2)*np.log((L+np.sqrt(L**2+a_val**2))/a_val) - (1/2)*(np.sqrt(L**2+a_val**2)-a_val)
        z_sing_integral_G = 2 * (I_sing(l1, a) / l1**2 + I_sing(l2, a) / l2**2)
        z_sing_integral = const * z_sing_integral_G
        dzdk_sing_integral = -z_sing_integral / k
        z_m_support = (zn[m], zn[m+2])
        def regular_kernel(z, zp):
            R = np.sqrt((z - zp)**2 + a**2)
            g_regular = (np.exp(-1j * k * R) - 1) / R if R > 1e-12 else -1j*k
            return self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * (g_regular + self._integrand_kernel(z, zp, k, a, singular_part_only=False)) * self._rooftop(zp, zn[m+1], dL[m], dL[m+1])
        z_reg_integral = dblquad(regular_kernel, z_m_support[0], z_m_support[1], z_m_support[0], z_m_support[1], epsrel=eps)[0]
        def regular_kernel_dk(z, zp):
             return self._rooftop(z, zn[m+1], dL[m], dL[m+1]) * self._integrand_kernel_dk(z, zp, k, a) * self._rooftop(zp, zn[m+1], dL[m], dL[m+1])
        dzdk_reg_integral = dblquad(regular_kernel_dk, z_m_support[0], z_m_support[1], z_m_support[0], z_m_support[1], epsrel=eps)[0]
        return z_reg_integral + z_sing_integral, dzdk_reg_integral + dzdk_sing_integral

    def _reconstruct_R_via_swe(self, k):
        zn, _, dL = self._create_geometry(self.cfg['Dipole']['Length'], self.cfg['Mesh']['Segments'])
        N_basis = self.cfg['Mesh']['Segments'] - 1
        n_max_limit = int(k * self.cfg['Dipole']['Length']) + 30; conv_tol = 1e-5
        R_swe = np.zeros((N_basis, N_basis), dtype=np.complex128); total_power_ref = 0.0
        I_ref = np.sin(np.pi * (zn[1:-1] + self.cfg['Dipole']['Length']/2) / self.cfg['Dipole']['Length'])
        for n in range(1, n_max_limit + 1):
            F_n = np.zeros(N_basis, dtype=np.complex128)
            for p in range(N_basis):
                integrand = lambda z: self._rooftop(z, zn[p+1], dL[p], dL[p+1]) * self._stable_bessel_integrand(n, k, z)
                try:
                    val, _ = quad(integrand, zn[p], zn[p+2], epsrel=self.cfg['Quadrature']['EpsRelNear'], limit=200)
                    F_n[p] = val
                except Exception as e:
                    warnings.warn(f"Integration failed for SWE n={n}, p={p}. Setting F_n to 0. Error: {e}", UserWarning)
                    F_n[p] = 0.0
            power_n_factor = (k**2 * self.eta0 / (2 * np.pi * n * (n+1)))
            R_n_contribution = power_n_factor * np.outer(F_n.conj(), F_n)
            R_swe += R_n_contribution
            power_n_ref = 0.5 * np.real(I_ref.conj().T @ R_n_contribution @ I_ref)
            total_power_ref += power_n_ref
            if total_power_ref > 1e-12 and np.abs(power_n_ref / total_power_ref) < conv_tol: break
        
        max_imag = np.max(np.abs(np.imag(R_swe))); max_real = np.max(np.abs(np.real(R_swe)))
        if self.cfg['Numerics']['EnforceRHermiticity'] and max_real > 0 and max_imag / max_real > 1e-8:
            warnings.warn(f"R_swe Hermiticity enforced. Max rel imag: {max_imag/max_real:.2e}", UserWarning)
            R_swe = (R_swe + R_swe.conj().T) / 2.0
        return np.real(R_swe)

    def _solve_and_filter_modes(self, X, R):
        try:
            vals, V = eigh(X, R)
        except np.linalg.LinAlgError:
            warnings.warn("eigh failed due to non-positive definite R. Applying regularization.", UserWarning)
            delta = self.cfg['Numerics']['RegularizationFactor']
            R_reg = R + np.eye(R.shape[0]) * delta * np.max(np.abs(R))
            try:
                vals, V = eigh(X, R_reg)
            except np.linalg.LinAlgError as e:
                raise CMAError("Eigensolver failed even after regularization.") from e

        idx = np.argsort(np.abs(vals))
        N_modes = min(self.cfg['Execution']['NumModes'], len(vals))
        vals, V = vals[idx][:N_modes], V[:, idx][:, :N_modes]
        U = np.zeros_like(V); valid_indices = []; filtered_indices = []
        power_rad_all = 0.5 * np.real(np.diag(V.conj().T @ R @ V))
        power_dominant = np.max(power_rad_all) if len(power_rad_all) > 0 else 0.0
        power_threshold = self.cfg['Execution']['PowerFilterThreshold'] * power_dominant if power_dominant > 1e-12 else 1e-12
        for i in range(N_modes):
            if power_rad_all[i] < power_threshold:
                filtered_indices.append(idx[i]); continue
            u_i = V[:, i].copy()
            for j in range(len(valid_indices)):
                u_j = U[:, valid_indices[j]]; dot_product = u_j.conj().T @ R @ u_i; u_i -= dot_product * u_j
            norm_factor = np.sqrt(np.abs(u_i.conj().T @ R @ u_i))
            if norm_factor > 1e-9:
                U[:, i] = u_i / norm_factor; valid_indices.append(i)
            else: filtered_indices.append(idx[i])
        return vals[valid_indices], U[:, valid_indices], filtered_indices

    def calculate_radiation_pattern(self, k, J_n, theta_rad=None):
        if theta_rad is None:
            # *** BUG FIX IS HERE ***
            # The default number of points is now 361 to match the analysis script.
            theta_rad = np.linspace(1e-6, np.pi - 1e-6, 361)
            
        zn, _, dL = self._create_geometry(self.cfg['Dipole']['Length'], self.cfg['Mesh']['Segments'])
        AF = np.zeros((J_n.shape[1], len(theta_rad)), dtype=np.complex128)
        
        # Vectorized calculation for performance
        z_vals = np.linspace(-self.cfg['Dipole']['Length']/2, self.cfg['Dipole']['Length']/2, 401)
        phase_term = np.exp(1j * k * z_vals[:, np.newaxis] * np.cos(theta_rad))
        
        for i in range(J_n.shape[1]):
            I_coeffs = J_n[:, i]
            current_dist = np.zeros_like(z_vals, dtype=np.complex128)
            for p in range(len(I_coeffs)):
                current_dist += I_coeffs[p] * self._rooftop(z_vals, zn[p+1], dL[p], dL[p+1])
            
            integrand_vals = current_dist[:, np.newaxis] * phase_term
            AF[i, :] = np.trapz(integrand_vals, z_vals, axis=0)
            
        return (-1j * k * self.eta0 / (4 * np.pi)) * AF * np.sin(theta_rad)

    @staticmethod
    def _stable_bessel_integrand(n, k, z):
        kz = k * z
        blend_center = n / 2.0; blend_width = n / 10.0
        weight = 0.5 * (1.0 - np.tanh((kz - blend_center) / blend_width))
        if weight > 0.999:
            if kz < 1e-9: return 0.0
            log_val = n * np.log(kz) - np.sum(np.log(np.arange(1, 2*n+2, 2)))
            jn_approx = np.exp(log_val)
            jn_deriv_approx = n * jn_approx / kz
            return (n * (n+1) / kz) * jn_approx + jn_deriv_approx
        
        scipy_val = (n * (n+1) / (kz + 1e-12)) * spherical_jn(n, kz) + spherical_jn(n, kz, derivative=True)
        if weight < 0.001: return scipy_val
        
        if kz < 1e-9: asymp_val = 0.0
        else:
            log_val = n * np.log(kz) - np.sum(np.log(np.arange(1, 2*n+2, 2)))
            jn_approx = np.exp(log_val)
            jn_deriv_approx = n * jn_approx / kz
            asymp_val = (n * (n+1) / kz) * jn_approx + jn_deriv_approx
        
        return weight * asymp_val + (1 - weight) * scipy_val
    
    # --- Other utility methods ---
    @staticmethod
    def _integrand_kernel(z, zp, k, a, singular_part_only=True):
        R = np.sqrt((z - zp)**2 + a**2);
        if R < 1e-12: return 0
        g = np.exp(-1j * k * R) / R
        if singular_part_only: return g
        g_deriv_term = (1j*k*R + 1) * g / R**2
        return (g + g_deriv_term / k**2)
    @staticmethod
    def _integrand_kernel_dk(z, zp, k, a):
        R = np.sqrt((z - zp)**2 + a**2);
        if R < 1e-12: return 0
        exp_term = np.exp(-1j * k * R)
        term1 = -1j * exp_term
        term2 = (-2/(k**3)) * (1 + 1j*k*R) * exp_term / R**3 + (-1j*R/k**2) * exp_term / R**3 + (1/k**2) * (1 + 1j*k*R) * (-1j*R) * exp_term / R**3
        return term1 + term2
    @staticmethod
    def _rooftop(z, c, d1, d2): return np.piecewise(z, [(z >= c-d1) & (z < c), (z >= c) & (z <= c+d2)], [lambda x:(x-(c-d1))/d1, lambda x:((c+d2)-x)/d2, 0.0])
    @staticmethod
    def _create_geometry(L, N): zn=np.linspace(-L/2,L/2,N+1); return zn, (zn[1:]+zn[:-1])/2, np.diff(zn)
    @staticmethod
    def validate_config(cfg):
        d = {
            'Dipole': {'Length': 0.5, 'Radius': 0.001},
            'Mesh': {'Segments': 51},
            'Feed': {'GapWidth': 0.005},
            'Execution': {'Frequency': 300e6, 'NumModes': 15, 'Verbose': False, 'PowerFilterThreshold': 1e-7, 'ModeTrackingThreshold': 0.9},
            'Quadrature': {'DuffyThresholdFactor': 2.5, 'EpsRelNear': 1e-9, 'EpsRelFar': 1e-7},
            'Numerics': {'EnforceRHermiticity': True, 'LambdaCutoff': 100, 'RegularizationFactor': 1e-12}
        }
        def merge(base, overlay):
            for k, v in overlay.items():
                if isinstance(v, dict) and k in base: base[k] = merge(base.get(k, {}), v)
                else: base[k] = v
            return base
        final_cfg = merge(d, cfg or {})
        if final_cfg['Mesh']['Segments'] % 2 != 1: raise ValueError("Segments must be odd.")
        return final_cfg
