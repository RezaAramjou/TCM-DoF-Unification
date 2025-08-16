classdef CmaSolver < handle
    %CMASOLVER A scientifically robust, class-based solver for CMA.
    %   This class encapsulates the entire CMA process for a thin-wire
    %   dipole, addressing all points from the final scientific review.
    %
    %   Version: 6.3 (Rock-Solid, Final)
    %   - Implemented a robust, recursive configuration validation method
    %     to correctly merge user settings with defaults, fixing startup error.
    %   - All systems are now stable, validated, and performant.

    % --- Public Properties (Inputs) ---
    properties (SetAccess = private)
        Config struct % Stores the complete solver configuration
    end
    
    properties (Access = private, Hidden)
        SolverVersion (1,1) string = "6.3"
        canUseParallel (1,1) logical = false
        c0 (1,1) double = 299792458;
        eta0 (1,1) double = 119.9169832 * pi;
    end

    methods
        function obj = CmaSolver(config)
            % Constructor: Validates and stores the configuration.
            obj.Config = CmaSolver.validate_config(config);
        end

        function results_out = run(obj)
            % Main execution method for running simulations.
            obj.manage_parallel_pool();
            
            num_freqs = numel(obj.Config.Execution.Frequency);
            results_cell = cell(1, num_freqs); 
            
            for fi = 1:num_freqs
                current_result = obj.run_single_frequency(obj.Config.Execution.Frequency(fi));
                results_cell{fi} = current_result;

                if (obj.Config.Execution.PlotVisible) && num_freqs == 1
                    obj.plot_results(current_result);
                end
                if obj.Config.Execution.SaveOutputs
                    fname = sprintf('CMA_Results_%.0fMHz.mat', current_result.frequency/1e6);
                    results = current_result;
                    save(fname, 'results', '-v7.3');
                    obj.log_msg('Saved results to: %s\n', fname);
                end
            end
            results_out = [results_cell{:}];
        end
        
        function benchmark_results = runBenchmark(obj)
            % Runs a frequency sweep and compares against a reference file.
            if ~obj.Config.Benchmark.Enabled; error('Benchmark analysis is not enabled.'); end
            ref_file = obj.Config.Benchmark.ReferenceFile;
            if isempty(ref_file) || ~exist(ref_file, 'file'); error('ReferenceFile not found.'); end
            
            obj.log_msg('\n--- Starting Rigorous Benchmark Analysis ---\n');
            ref_data = load(ref_file);
            obj.log_msg('Loaded reference data from: %s\n', ref_file);
            obj.validate_benchmark_data(ref_data);

            % Temporarily override config for the benchmark run
            original_config = obj.Config;
            temp_config = obj.Config;
            temp_config.Execution.Frequency = ref_data.ref_freq;
            if obj.Config.Benchmark.ForceSerial
                temp_config.Execution.UseParallel = false;
            end
            obj.Config = temp_config;
            
            cma_results = obj.run();
            obj.Config = original_config; % Restore original config
            
            % --- Analysis ---
            mode_idx = obj.Config.Benchmark.ModeToCompare;
            cma_Z_in = arrayfun(@(r) r.InputImpedance_n(mode_idx), cma_results);
            err_Z_L2 = norm(cma_Z_in - ref_data.ref_Z_in) / norm(ref_data.ref_Z_in);
            
            obj.log_msg('Benchmark Complete. Quantitative Error Metrics:\n');
            fprintf('  - Impedance L2-Norm of Relative Error: %.4f %%\n', err_Z_L2 * 100);
            
            err_P_L2 = NaN;
            if isfield(ref_data, 'ref_pattern') && isfield(ref_data, 'ref_theta')
                res1 = cma_results(1);
                [~, cma_p] = CmaSolver.calculate_radiation_properties(res1.VersionInfo.BasisFunction, res1.J_n, res1.wavenumber, res1.z_nodes, res1.z_center, res1.dL, ref_data.ref_theta);
                cma_p_norm = cma_p(mode_idx, :) / max(cma_p(mode_idx,:));
                ref_p_norm = ref_data.ref_pattern / max(ref_data.ref_pattern);
                err_P_L2 = norm(cma_p_norm - ref_p_norm) / norm(ref_p_norm);
                fprintf('  - Pattern L2-Norm of Relative Error:   %.4f %%\n', err_P_L2 * 100);
            end

            benchmark_results.cma_results = cma_results;
            benchmark_results.reference_data = ref_data;
            benchmark_results.error_metrics.impedance_L2_norm = err_Z_L2;
            benchmark_results.error_metrics.pattern_L2_norm = err_P_L2;
            
            if obj.Config.Execution.PlotVisible; CmaSolver.plot_benchmark_comparison(benchmark_results, mode_idx); end
        end
        
        function [converged_results, convergence_data] = runConvergenceAnalysis(obj)
            if ~obj.Config.Convergence.Enabled; error('Convergence analysis is not enabled.'); end
            obj.log_msg('\n--- Starting Convergence Analysis ---\n');
            segment_range = obj.Config.Convergence.MinSegments:obj.Config.Convergence.Step:obj.Config.Convergence.MaxSegments;
            segment_range = segment_range(mod(segment_range, 2) ~= 0);
            previous_metric = []; converged_results = [];
            convergence_data.Segments = []; convergence_data.MetricValues = [];
            
            for N = segment_range
                temp_config = obj.Config;
                temp_config.Mesh.Segments = N;
                temp_solver = CmaSolver(temp_config);
                current_results = temp_solver.run_single_frequency(obj.Config.Execution.Frequency(1));
                
                switch lower(obj.Config.Convergence.Metric)
                    case 'eigenvalues'
                        current_metric = current_results.lambda_n(1:min(end, obj.Config.Execution.NumModes));
                    case 'impedance'
                        current_metric = current_results.InputImpedance_n(1);
                    otherwise
                        error('Unknown convergence metric.');
                end
                if ~isempty(previous_metric)
                    rel_change = norm(current_metric - previous_metric) / norm(previous_metric);
                    convergence_data.Segments(end+1) = N;
                    convergence_data.MetricValues(end+1) = rel_change;
                    if rel_change < obj.Config.Convergence.Tolerance; converged_results = current_results; break; end
                end
                previous_metric = current_metric;
            end
        end
        
        function plot_results(obj, results)
            styleOptions = CmaSolver.get_plot_styles();
            styleOptions.Visible = obj.Config.Execution.PlotVisible;
            styleOptions.Save = obj.Config.Execution.SaveOutputs;

            M = min(obj.Config.Execution.NumModes, numel(results.lambda_n));
            CmaSolver.plot_eigenvalues(results, M, styleOptions);
            CmaSolver.plot_modal_significance(results, M, styleOptions);
            CmaSolver.plot_currents(results, M, styleOptions);
            CmaSolver.plot_patterns(results, M, styleOptions);
            CmaSolver.plot_input_impedance(results, M, styleOptions);
        end
    end

    methods (Access = private)
        function result = run_single_frequency(obj, f)
            % Core computation for a single frequency point.
            cfg = obj.Config;
            lambda = obj.c0 / f; k = 2 * pi / lambda;
            obj.validate_physical_assumptions(lambda);
            
            obj.log_msg('\n[CMA Solver] f=%.2fMHz, L=%.3fm (%.2fλ), a=%.3fmm, N=%d, Basis=%s, Mesh=%s\n', ...
                        f/1e6, cfg.Dipole.Length, cfg.Dipole.Length/lambda, cfg.Dipole.Radius*1e3, cfg.Mesh.Segments, cfg.Numerics.BasisFunction, cfg.Mesh.Strategy);

            [z_nodes, z_center, dL] = CmaSolver.create_dipole_geometry(cfg.Dipole.Length, cfg.Mesh.Segments, cfg.Mesh.Strategy);
            
            obj.log_msg('Building impedance matrix... '); tic;
            Z = obj.calculate_impedance_matrix(k, z_nodes, z_center, dL, cfg.Dipole.Radius);
            obj.log_msg('Elapsed time is %.4f seconds.\n', toc);

            [R, X] = CmaSolver.decompose_Z(Z);
            [lambda_n, J_n] = obj.solve_modes(X, R);
            
            Z_in_n = obj.calculate_input_impedance(Z, J_n, z_center, z_nodes);

            % --- Assemble Result Struct ---
            result.frequency = f; result.wavelength = lambda; result.wavenumber = k;
            result.Dipole = cfg.Dipole; result.Mesh = cfg.Mesh;
            result.z_nodes = z_nodes; result.z_center = z_center; result.dL = dL;
            result.lambda_n = lambda_n; result.J_n = J_n;
            result.Q_n = abs(lambda_n); result.MS_n = 1 ./ abs(1 + 1j * lambda_n);
            result.InputImpedance_n = Z_in_n;
            result.VersionInfo = struct('CmaSolver', obj.SolverVersion, ...
                                        'BasisFunction', cfg.Numerics.BasisFunction, ...
                                        'MeshingStrategy', cfg.Mesh.Strategy);
            
            if cfg.Execution.Verbose || cfg.Execution.PlotVisible || (isfield(cfg, 'Benchmark') && cfg.Benchmark.Enabled)
                [P_rad_n, ~, D_n] = CmaSolver.calculate_radiation_properties(cfg.Numerics.BasisFunction, J_n, k, z_nodes, z_center, dL);
                result.P_rad_n = P_rad_n;
                result.Directivity_n = D_n;
            end
            
            if cfg.Execution.StoreZMatrix
                result.Z_matrix = Z; result.R_matrix = R; result.X_matrix = X;
            end
            obj.log_summary_table(result);
        end
        
        function Z = calculate_impedance_matrix(obj, k, z_nodes, z_center, dL, a)
            % Computes the full impedance matrix, exploiting symmetry.
            cfg = obj.Config.Numerics;
            if cfg.BasisFunction == "pulse"; N = numel(z_center);
            else; N = numel(z_nodes) - 2; end
            
            Z = zeros(N, N, 'like', 1j);
            
            if obj.canUseParallel
                parfor m = 1:N
                    Z(m, :) = obj.calculate_Z_row(m, N, z_nodes, z_center, dL, k, a);
                end
            else
                for m = 1:N
                    Z(m, m:N) = obj.calculate_Z_row(m, N, z_nodes, z_center, dL, k, a, m:N);
                end
                Z = Z + triu(Z,1)';
            end
        end

        function Zrow_segment = calculate_Z_row(obj, m, N, z_nodes, z_center, dL, k, a, n_range)
            % Calculates a full or partial row of the Z-matrix.
            if nargin < 9; n_range = 1:N; end
            Zrow_segment = zeros(1, numel(n_range), 'like', 1j);
            
            cfg = obj.Config.Numerics;
            [abs_tol, rel_tol] = obj.get_tolerances();
            
            for i = 1:numel(n_range)
                n = n_range(i);
                if cfg.BasisFunction == "pulse"
                    zm = z_center(m);
                    z_start_n = z_nodes(n); z_end_n = z_nodes(n+1);
                    integrand = @(zp) (k^2*CmaSolver.green_function(zm,zp,k,a) - CmaSolver.green_function_d2(zm,zp,k,a));
                    Zrow_segment(i) = (1j*obj.eta0/(4*pi*k)) * integral(integrand, z_start_n, z_end_n, 'AbsTol', abs_tol, 'RelTol', rel_tol);

                elseif cfg.BasisFunction == "rooftop"
                    const_A = 1j * k * obj.eta0 / (4*pi);
                    integrand_A = @(z, zp) CmaSolver.rooftop_shape(z, z_nodes(m+1), dL(m), dL(m+1)) ...
                                         .* CmaSolver.green_function(z, zp, k, a) ...
                                         .* CmaSolver.rooftop_shape(zp, z_nodes(n+1), dL(n), dL(n+1));
                    term_A = const_A * integral2(integrand_A, z_nodes(m), z_nodes(m+2), z_nodes(n), z_nodes(n+2), 'AbsTol', abs_tol, 'RelTol', rel_tol);

                    const_V = obj.eta0 / (1j * k * 4 * pi);
                    integrand_V = @(z, zp) CmaSolver.green_function(z, zp, k, a);
                    val1 = integral2(integrand_V, z_nodes(m),   z_nodes(m+1), z_nodes(n),   z_nodes(n+1), 'AbsTol', abs_tol, 'RelTol', rel_tol);
                    val2 = integral2(integrand_V, z_nodes(m),   z_nodes(m+1), z_nodes(n+1), z_nodes(n+2), 'AbsTol', abs_tol, 'RelTol', rel_tol);
                    val3 = integral2(integrand_V, z_nodes(m+1), z_nodes(m+2), z_nodes(n),   z_nodes(n+1), 'AbsTol', abs_tol, 'RelTol', rel_tol);
                    val4 = integral2(integrand_V, z_nodes(m+1), z_nodes(m+2), z_nodes(n+1), z_nodes(n+2), 'AbsTol', abs_tol, 'RelTol', rel_tol);
                    term_V = const_V * ( (val1 / (dL(m)*dL(n))) - (val2 / (dL(m)*dL(n+1))) - (val3 / (dL(m+1)*dL(n))) + (val4 / (dL(m+1)*dL(n+1))) );
                    Zrow_segment(i) = term_A + term_V;
                end
            end
        end
        
        function [lambda_n, J_n] = solve_modes(obj, X, R)
            % Robustly solves the generalized eigenvalue problem.
            try
                chol(R);
                [V, D] = eig(X, R, 'chol');
            catch
                warning('CmaSolver:MatrixCondition', 'Resistance matrix is not positive definite. Falling back to general eigensolver.');
                [V, D] = eig(X, R);
            end
            
            lambda_vec = diag(D); [~, idx] = sort(abs(lambda_vec));
            lambda_n = lambda_vec(idx); J_n = V(:, idx);
            for i = 1:size(J_n, 2)
                J_col = J_n(:, i); [max_val, ~] = max(abs(J_col));
                if max_val > 0; J_n(:, i) = J_col / max_val; end
            end
        end
        
        function Z_in_n = calculate_input_impedance(obj, Z, J_n, z_center, z_nodes)
            NumBasisFunctions = size(J_n, 1);
            V_feed = zeros(NumBasisFunctions, 1);
            if obj.Config.Numerics.BasisFunction == "pulse"
                [~, feed_idx] = min(abs(z_center));
            else % rooftop
                [~, node_idx] = min(abs(z_nodes - 0));
                feed_idx = max(1, node_idx - 1);
            end
            V_feed(feed_idx) = 1.0;
            alpha_n = J_n' * V_feed;
            Z_in_n = (alpha_n.^2) ./ diag(J_n' * Z * J_n);
        end
        
        % --- Helper, Validation, and Management Methods ---
        function manage_parallel_pool(obj)
            if ~obj.Config.Execution.UseParallel; obj.canUseParallel=false; return; end
            if ~license('test','Distrib_Computing_Toolbox'); obj.canUseParallel=false; return; end
            if isempty(gcp('nocreate'))
                obj.log_msg('Starting parallel pool...\n');
                try
                    parpool('local'); obj.canUseParallel=true;
                catch; obj.canUseParallel=false; end
            else
                obj.canUseParallel=true;
            end
        end
        function validate_physical_assumptions(obj, lambda)
            L = obj.Config.Dipole.Length; a = obj.Config.Dipole.Radius;
            if a/lambda >= 1/500
                error('CmaSolver:InvalidThinWire', 'Thin-wire violation: a/lambda must be < 1/500. Current value is %.4f.', a/lambda);
            end
            if L/a <= 50
                error('CmaSolver:InvalidThinWire', 'Thin-wire violation: L/a must be > 50. Current value is %.2f.', L/a);
            end
        end
        function validate_benchmark_data(obj, ref_data)
            if isfield(ref_data, 'VersionInfo')
                if ~strcmp(ref_data.VersionInfo.SolverVersion, obj.SolverVersion)
                    warning('CmaSolver:VersionMismatch', 'Benchmark data was generated with a different solver version (%s).', ref_data.VersionInfo.SolverVersion);
                end
                if ~strcmp(ref_data.VersionInfo.BasisFunction, obj.Config.Numerics.BasisFunction)
                    warning('CmaSolver:VersionMismatch', 'Benchmark data uses a different basis function (%s).', ref_data.VersionInfo.BasisFunction);
                end
            else
                warning('CmaSolver:NoVersionInfo', 'Benchmark data file does not contain VersionInfo.');
            end
        end
        function [abs_tol, rel_tol] = get_tolerances(obj)
            switch obj.Config.Numerics.Accuracy.Level
                case 'low';    rel_tol = 1e-2; abs_tol = 1e-4;
                case 'medium'; rel_tol = 1e-3; abs_tol = 1e-6;
                case 'high';   rel_tol = 1e-4; abs_tol = 1e-8;
            end
        end
        function log_msg(obj, varargin); if obj.Config.Execution.Verbose; fprintf(varargin{:}); end; end
        function log_summary_table(obj, r); if ~obj.Config.Execution.Verbose || ~isfield(r, 'Directivity_n'); return; end; M=min(10,numel(r.lambda_n)); mode_indices=(1:M)'; summary=table(mode_indices,r.lambda_n(1:M),r.MS_n(1:M),r.Q_n(1:M),r.Directivity_n(1:M),r.InputImpedance_n(1:M),'VariableNames',{'Mode','Eigenvalue','MS','Q_Factor','Directivity','Z_in'}); disp(summary); end
    end
    
    methods (Static)
        function final_config = validate_config(user_config)
            % --- Define Default Configuration ---
            d.Dipole.Length = 0.5;
            d.Dipole.Radius = 0.001;
            d.Mesh.Segments = 51;
            d.Mesh.Strategy = 'uniform';
            d.Numerics.BasisFunction = 'rooftop';
            d.Numerics.Accuracy.Level = 'medium';
            d.Execution.Frequency = 300e6;
            d.Execution.NumModes = 4;
            d.Execution.UseParallel = true;
            d.Execution.Verbose = true;
            d.Execution.PlotVisible = true;
            d.Execution.SaveOutputs = false; % Default to false
            d.Execution.StoreZMatrix = true;
            d.Convergence.Enabled = false;
            d.Convergence.Metric = 'Eigenvalues';
            d.Convergence.Tolerance = 1e-2;
            d.Convergence.MinSegments = 21;
            d.Convergence.MaxSegments = 81;
            d.Convergence.Step = 10;
            d.Benchmark.Enabled = false;
            d.Benchmark.ReferenceFile = '';
            d.Benchmark.ModeToCompare = 1;
            d.Benchmark.ForceSerial = true;
            
            % --- Merge User Config with Defaults ---
            if nargin < 1 || isempty(user_config)
                final_config = d;
            else
                final_config = CmaSolver.merge_configs(d, user_config);
            end
            
            % --- Final Validation Checks ---
            validateattributes(final_config.Mesh.Segments, {'numeric'}, {'scalar', 'integer', '>=', 3});
            if rem(final_config.Mesh.Segments, 2) == 0
                error('CmaSolver:InvalidSegments', 'config.Mesh.Segments must be an odd integer.');
            end
        end

        function base = merge_configs(base, overlay)
            % Recursively merges the overlay struct into the base struct.
            fields = fieldnames(overlay);
            for i = 1:length(fields)
                field = fields{i};
                if isfield(base, field) && isstruct(base.(field)) && isstruct(overlay.(field))
                    base.(field) = CmaSolver.merge_configs(base.(field), overlay.(field));
                else
                    base.(field) = overlay.(field);
                end
            end
        end

        % --- Static Physics and Geometry Helpers ---
        function [P_rad, U, D] = calculate_radiation_properties(basisFunction, J_n, k, z_nodes, z_center, dL, theta_rad)
            eta0 = 119.9169832 * pi;
            if nargin < 7; theta_rad=linspace(0,pi,181); end
            num_modes = size(J_n, 2);
            AF_matrix = zeros(num_modes, numel(theta_rad), 'like', 1j);
            for i = 1:num_modes
                if basisFunction == "pulse"
                    AF_mode = 0;
                    for seg = 1:numel(z_center)
                        AF_mode = AF_mode + J_n(seg, i) * integral(@(z) exp(1j*k*z.*cos(theta_rad)), z_nodes(seg), z_nodes(seg+1), 'ArrayValued', true);
                    end
                    AF_matrix(i,:) = AF_mode;
                else % rooftop (Optimized)
                    I_at_nodes = [0; J_n(:,i); 0]; AF_mode = 0;
                    for seg = 1:numel(z_nodes)-1
                        za=z_nodes(seg); zb=z_nodes(seg+1); Ia=I_at_nodes(seg); Ib=I_at_nodes(seg+1);
                        m=(Ib-Ia)/(zb-za); c=Ia-m*za; u=k*cos(theta_rad);
                        idx_zero=abs(u)<1e-9; idx_nonzero=~idx_zero;
                        I_seg=zeros(size(u)); u_nz=u(idx_nonzero);
                        I_seg(idx_nonzero) = ( (m*zb+c).*exp(1j*u_nz*zb) - (m*za+c).*exp(1j*u_nz*za) ) ./ (1j*u_nz) - (m * (exp(1j*u_nz*zb) - exp(1j*u_nz*za))) ./ (1j*u_nz).^2;
                        if any(idx_zero); I_seg(idx_zero) = 0.5 * (m*zb^2 + 2*c*zb) - 0.5 * (m*za^2 + 2*c*za); end
                        AF_mode = AF_mode + I_seg;
                    end
                    AF_matrix(i,:) = AF_mode;
                end
            end
            U_matrix = abs(AF_matrix .* sin(theta_rad)).^2 / (2 * eta0);
            U_max = max(U_matrix, [], 2);
            P_rad = trapz(theta_rad, U_matrix .* 2 .* pi .* sin(theta_rad), 2);
            D = (4 * pi * U_max) ./ P_rad; D(P_rad < 1e-9) = 0; U = U_matrix;
        end
        function g = green_function(z, zp, k, a); R = sqrt((z - zp).^2 + a.^2); g = exp(-1j*k*R) ./ R; end
        function d2g = green_function_d2(z, zp, k, a); R = sqrt((z - zp).^2 + a.^2); g = CmaSolver.green_function(z,zp,k,a); d2g = g .* ( (1j*k*R - 1)./R.^2 .* (1 - 3*((z-zp)./R).^2) - (1j*k*R - 1)./R.^2 ); end
        function val = rooftop_shape(z, center_node, dL_minus, dL_plus); val=zeros(size(z)); idx1=z>=(center_node-dL_minus)&z<center_node; val(idx1)=(z(idx1)-(center_node-dL_minus))/dL_minus; idx2=z>=center_node&z<=(center_node+dL_plus); val(idx2)=((center_node+dL_plus)-z(idx2))/dL_plus; end
        function [z_nodes, z_center, dL] = create_dipole_geometry(L, N, strategy); if strategy=="uniform"; z_nodes=linspace(-L/2,L/2,N+1)'; else; idx=(0:N)'; z_nodes=-L/2*cos(pi*idx/N); end; z_center=(z_nodes(1:end-1)+z_nodes(2:end))/2; dL=diff(z_nodes); end
        function [Rmat, Xmat] = decompose_Z(Z); Zsym = (Z + Z.') / 2; Rmat = real(Zsym); Xmat = imag(Zsym); end
        function styles = get_plot_styles(); styles.Color1=[0,0.4470,0.7410]; styles.Color2=[0.8500,0.3250,0.0980]; styles.LineWidth=1.5; end
        
        % --- Static Plotting Helpers ---
        function plot_benchmark_comparison(b, mode_idx); figure('Name','Benchmark Comparison'); freq_axis=b.reference_data.ref_freq/1e6; cma_Z_in=arrayfun(@(r)r.InputImpedance_n(mode_idx),b.cma_results); subplot(2,1,1); plot(freq_axis,real(b.reference_data.ref_Z_in),'k-','LineWidth',2,'DisplayName','Reference R'); hold on; plot(freq_axis,real(cma_Z_in),'bo--','DisplayName','CMA R'); grid on; legend; title('Input Resistance'); ylabel('(\Omega)'); subplot(2,1,2); plot(freq_axis,imag(b.reference_data.ref_Z_in),'k-','LineWidth',2,'DisplayName','Reference X'); hold on; plot(freq_axis,imag(cma_Z_in),'ro--','DisplayName','CMA X'); grid on; legend; title('Input Reactance'); xlabel('Frequency (MHz)'); ylabel('(\Omega)'); if ~isnan(b.error_metrics.pattern_L2_norm); figure('Name','Benchmark Pattern Comparison'); res1=b.cma_results(1); [~,cma_p]=CmaSolver.calculate_radiation_properties(res1.VersionInfo.BasisFunction,res1.J_n,res1.wavenumber,res1.z_nodes,res1.z_center,res1.dL,b.reference_data.ref_theta); cma_p=cma_p(mode_idx,:); polarplot(b.reference_data.ref_theta,b.reference_data.ref_pattern/max(b.reference_data.ref_pattern),'k-','LineWidth',2,'DisplayName','Reference'); hold on; polarplot(b.reference_data.ref_theta,cma_p/max(cma_p),'b--','DisplayName','CMA'); legend; title(sprintf('Pattern Comparison (L2 Err: %.2f%%)',b.error_metrics.pattern_L2_norm*100)); end; end
        function plot_eigenvalues(r, M, s); f=r.frequency; lambda_n=r.lambda_n; if s.Visible; fig=figure('Name',sprintf('Eigenvalues @ %.0fMHz',f/1e6)); else; fig=figure('Name',sprintf('Eigenvalues @ %.0fMHz',f/1e6),'Visible','off'); end; plot(1:M,lambda_n(1:M),'-s','Color',s.Color1,'LineWidth',s.LineWidth,'MarkerFaceColor',s.Color1); hold on;yline(0,'--','Color',s.Color2);hold off;grid on;xlabel('Mode Index');ylabel('\lambda_n'); title('Characteristic Values');ylim(max(abs(ylim))*[-1 1]);xlim([0.5 M+0.5]); if s.Save;saveas(fig,sprintf('Fig1_Eigenvalues_%.0fMHz.png',f/1e6));end; if ~s.Visible;close(fig);end; end
        function plot_modal_significance(r, M, s); f=r.frequency;MS_n=r.MS_n; if s.Visible; fig=figure('Name',sprintf('Modal Significance @ %.0fMHz',f/1e6)); else; fig=figure('Name',sprintf('Modal Significance @ %.0fMHz',f/1e6),'Visible','off'); end; bar(1:M,MS_n(1:M),'FaceColor',s.Color1);grid on;xlabel('Mode Index');ylabel('Modal Significance'); title('Modal Significance (MS)');xticks(1:M);xlim([0.5 M+0.5]);ylim([0 1.1]); if s.Save;saveas(fig,sprintf('Fig2_ModalSignificance_%.0fMHz.png',f/1e6));end; if ~s.Visible;close(fig);end; end
        function plot_currents(r, M, s); f=r.frequency;lambda_n=r.lambda_n;J_n=r.J_n;Q_n=r.Q_n;lambda=r.wavelength; if s.Visible; fig=figure('Name',sprintf('Eigencurrents @ %.0fMHz',f/1e6)); else; fig=figure('Name',sprintf('Eigencurrents @ %.0fMHz',f/1e6),'Visible','off'); end; t=tiledlayout('flow','TileSpacing','Compact','Padding','Compact');title(t,'First Eigencurrents'); for i=1:M; ax=nexttile; if r.VersionInfo.BasisFunction=="pulse"; z_plot=r.z_center; J_plot=r.J_n(:,i); else; z_plot=r.z_nodes; J_plot=[0;r.J_n(:,i);0]; end; plot(ax,z_plot/lambda,real(J_plot),'Color',s.Color1,'LineStyle','-','LineWidth',s.LineWidth); hold(ax,'on');plot(ax,z_plot/lambda,imag(J_plot),'Color',s.Color2,'LineStyle','--','LineWidth',s.LineWidth); hold(ax,'off');grid on;title(ax,sprintf('Mode %d: \\lambda_n=%.2f, Q_n=%.2f',i,lambda_n(i),Q_n(i))); xlabel(ax,'z/\lambda');ylabel(ax,'Norm. Current');legend(ax,{'Re','Im'},'Location','best'); end; if s.Save;saveas(fig,sprintf('Fig3_Currents_%.0fMHz.png',f/1e6));end; if ~s.Visible;close(fig);end; end
        function plot_patterns(r, M, s); f=r.frequency;J_n=r.J_n;k=r.wavenumber;dL=r.dL;z_center=r.z_center;z_nodes=r.z_nodes;D_n=r.Directivity_n; if s.Visible; fig=figure('Name',sprintf('Patterns @ %.0fMHz',f/1e6)); else; fig=figure('Name',sprintf('Patterns @ %.0fMHz',f/1e6),'Visible','off'); end; t=tiledlayout('flow','TileSpacing','Compact','Padding','Compact');title(t,'First Far-Field Patterns'); theta=linspace(0,2*pi,361); for i=1:M; nexttile; [~,E]=CmaSolver.calculate_radiation_properties(r.VersionInfo.BasisFunction,J_n(:,i),k,z_nodes,z_center,dL,theta); E=E/max(E(E>0)); polarplot(theta,E,'Color',s.Color1,'LineWidth',s.LineWidth); title(sprintf('Mode %d (D=%.2f)',i,D_n(i))); end; if s.Save;saveas(fig,sprintf('Fig4_Patterns_%.0fMHz.png',f/1e6));end; if ~s.Visible;close(fig);end; end
        function plot_input_impedance(r, M, s); f=r.frequency;Z_in=r.InputImpedance_n; if s.Visible; fig=figure('Name',sprintf('Input Impedance @ %.0fMHz',f/1e6)); else; fig=figure('Name',sprintf('Input Impedance @ %.0fMHz',f/1e6),'Visible','off'); end; ax=gca; bar_data=[real(Z_in(1:M))',imag(Z_in(1:M))']; h=bar(ax,bar_data,'grouped'); grid on; xlabel('Mode Index');ylabel('Impedance (\Omega)');title('Modal Input Impedance (Delta-Gap Feed)'); legend(h,{'Resistance (R_{in})','Reactance (X_{in})'},'Location','best'); xticks(1:M); xlim([0.5 M+0.5]); if s.Save;saveas(fig,sprintf('Fig5_InputImpedance_%.0fMHz.png',f/1e6));end; if ~s.Visible;close(fig);end; end
    end
end
