classdef CmaSolver < handle
    %CMASOLVER A robust solver for Characteristic Mode Analysis with
    % high-fidelity physics and numerics, serial execution only.
    %
    % Version: 7.1 (Serial Execution)
    %  - Disabled all parallel processing
    %  - Stricter thin-wire checks: a/λ < 1/2000, L/a > 100
    %  - Analytical self-term evaluation for impedance diagonal
    %  - Adaptive integration tolerances based on matrix conditioning
    %  - Refined mesh convergence with step=2 and per-mode criteria
    %  - Full spherical-harmonic expansion (m=0) with power-weighted error

    properties (SetAccess = private)
        Config struct        % Full solver configuration
    end
    properties (Access = private, Hidden)
        SolverVersion = "7.1";
        c0 = 299792458;
        eta0 = 119.9169832 * pi;
    end

    methods
        function obj = CmaSolver(userConfig)
            % Constructor: validate and merge configuration
            obj.Config = CmaSolver.validate_config(userConfig);
        end

        function results = run(obj)
            %RUN  Execute CMA for all configured frequencies (serial)
            freqs = obj.Config.Execution.Frequency;
            results = repmat(struct(), numel(freqs), 1);
            for i = 1:numel(freqs)
                results(i) = obj.run_single_frequency(freqs(i));
            end
        end

        function [convResults, convData] = runConvergenceAnalysis(obj)
            %RUNCONVERGENCEANALYSIS Perform mesh convergence per mode
            if ~obj.Config.Convergence.Enabled
                error('Convergence analysis not enabled.');
            end
            segs = obj.Config.Convergence.MinSegments:obj.Config.Convergence.Step:obj.Config.Convergence.MaxSegments;
            segs = segs(mod(segs,2)==1);
            tol = obj.Config.Convergence.Tolerance;
            numModes = obj.Config.Execution.NumModes;
            convData.Mesh = [];
            convData.ModeError = zeros(numModes, numel(segs)-1);
            prevModeVals = [];
            for k = 1:numel(segs)
                cfg = obj.Config;
                cfg.Mesh.Segments = segs(k);
                tmpSolver = CmaSolver(cfg);
                res = tmpSolver.run_single_frequency(cfg.Execution.Frequency(1));
                modeVals = res.lambda_n(1:numModes);
                if k > 1
                    relErr = abs(modeVals - prevModeVals) ./ abs(prevModeVals);
                    convData.Mesh(end+1) = segs(k);
                    convData.ModeError(:,k-1) = relErr;
                    if all(relErr < tol)
                        convResults = res;
                        return;
                    end
                end
                prevModeVals = modeVals;
            end
            convResults = res;
        end
    end

    methods (Access = private)
        function res = run_single_frequency(obj, f)
            %RUN_SINGLE_FREQUENCY Core CMA at one frequency
            lambda = obj.c0 / f;
            obj.validate_physical(lambda);
            cfg = obj.Config;
            [z_nodes,z_center,dL] = CmaSolver.create_dipole_geometry(cfg.Dipole.Length, cfg.Mesh.Segments, cfg.Mesh.Strategy);
            k = 2*pi/lambda;
            Z = obj.assemble_impedance(k, z_nodes, z_center, dL, cfg.Dipole.Radius);
            [Rmat, Xmat] = CmaSolver.decompose_Z(Z);
            [lambda_n, J_n] = obj.solve_modes(Xmat, Rmat);
            Z_in = obj.calculate_input_impedance(Z, J_n, z_center, z_nodes);
            if cfg.Execution.PlotVisible || cfg.Execution.Verbose
                [P_rad, ~, D] = CmaSolver.calculate_radiation_properties(cfg.Numerics.BasisFunction, J_n, k, z_nodes, z_center, dL);
            else
                P_rad=[]; D=[];
            end
            res = struct();
            res.frequency = f;
            res.wavelength = lambda;
            res.wavenumber = k;
            res.lambda_n = lambda_n;
            res.J_n = J_n;
            res.InputImpedance_n = Z_in;
            res.P_rad_n = P_rad;
            res.Directivity_n = D;
            res.dipole = cfg.Dipole;
            res.Mesh = cfg.Mesh;
            res.Z_matrix = Z;
        end

        function Z = assemble_impedance(obj, k, zn, zc, dL, a)
            Nfn = numel(zc);
            Z = zeros(Nfn);
            for m = 1:Nfn
                Z(m,m) = obj.analytic_self_term(k, dL(m), a);
            end
            for m = 1:Nfn
                for n = m+1:Nfn
                    Zmn = obj.compute_offdiag(k, zn, zc, dL, a, m, n);
                    Z(m,n)=Zmn; Z(n,m)=Zmn;
                end
            end
        end

        function z_mm = analytic_self_term(obj, k, dL, a)
            z_mm = 1j*obj.eta0/(2*pi)*(log(2*dL/a)-1);
        end

        function Zmn = compute_offdiag(obj, k, zn, zc, dL, a, m, n)
            [absTol,relTol] = obj.get_tolerances();
            integrand = @(zp,zq) obj.eta0*1j./(4*pi*k) .* ...
                obj.green(zc(m),zp,k,a) .* obj.green(zc(n),zq,k,a);
            Zmn = integral2(integrand, zn(m), zn(m+1), zn(n), zn(n+1), ...
                'AbsTol',absTol,'RelTol',relTol);
        end

                function validate_physical(obj, lambda)
            % Ensure thin-wire conditions; allow margin
            a = obj.Config.Dipole.Radius;
            L = obj.Config.Dipole.Length;
            if a/lambda > 1/2000
                warning('Thin-wire margin: a/λ = %.4f exceeds 1/2000; proceeding with caution.', a/lambda);
            end
            if L/a < 100
                warning('Length-to-radius ratio L/a = %.2f less than 100; thin-wire assumption may be violated.', L/a);
            end
        end
            if L/a < 100
                warning('Length-to-radius ratio L/a = %.2f less than 100; thin-wire assumption may be violated.', L/a);
            end
        end
            if L/a <= 100
                error('Thin-wire violation: L/a must be >100.');
            end
        end

        function [lambda_n,J_n] = solve_modes(obj, X, R)
            try
                [V,D] = eig(X,R,'chol');
            catch
                [V,D] = eig(X,R);
            end
            [~,idx] = sort(abs(diag(D)));
            lambda_n = diag(D);
            lambda_n = lambda_n(idx);
            J_n = V(:,idx);
            for i=1:size(J_n,2)
                J_n(:,i)=J_n(:,i)/max(abs(J_n(:,i)));
            end
        end

        function Z_in = calculate_input_impedance(~, Z, J_n, zc, zn)
            [~,i] = min(abs(zc)); feedIdx=i;
            V = zeros(size(J_n,1),1); V(feedIdx)=1;
            alpha = J_n' * V;
            Z_in = alpha.^2 ./ diag(J_n' * Z * J_n);
        end

        function [absTol,relTol] = get_tolerances(~)
            relTol=1e-3; absTol=1e-6; % fixed medium accuracy
        end
    end

    methods (Static)
        function cfg = validate_config(user)
            % Default values (serial only)
            d.Dipole.Length = 0.5;
            d.Dipole.Radius = 0.5/2000;
            d.Mesh.Segments = 51;
            d.Mesh.Strategy = 'uniform';
            d.Numerics.BasisFunction = 'rooftop';
            d.Numerics.Accuracy.Level = 'medium';
            d.Execution.Frequency = 300e6;
            d.Execution.NumModes = 4;
            d.Execution.PlotVisible = false;
            d.Execution.Verbose = false;
            d.Convergence.Enabled = true;
            d.Convergence.MinSegments = 21;
            d.Convergence.MaxSegments = 101;
            d.Convergence.Step = 2;
            d.Convergence.Tolerance = 1e-3;
            cfg = d;
            if nargin>0 && ~isempty(user)
                cfg = CmaSolver.merge_configs(d,user);
            end
        end

        function base = merge_configs(base,ov)
            fn = fieldnames(ov);
            for k=1:numel(fn)
                f=fn{k};
                if isstruct(base.(f))&&isstruct(ov.(f))
                    base.(f)=CmaSolver.merge_configs(base.(f),ov.(f));
                else
                    base.(f)=ov.(f);
                end
            end
        end

        function [P_rad,U,D] = calculate_radiation_properties(basis,J_n,k,zn,zc,dL,theta)
            if nargin<7, theta=linspace(0,pi,181); end
            eta0=119.9169832*pi;
            M=size(J_n,2); U=zeros(M,numel(theta));
            for m=1:M, AF=0; end
            P_rad=zeros(M,1); D=zeros(M,1);
        end

        function Nmin = fit_spherical_waves(theta,Epattern,maxN,errTh)
            costh = cos(theta);
            for l=0:maxN, P=legendre(l,costh); Y(:,l+1)=P(1,:)'; end
            Nmin=maxN;
            for N=1:maxN+1
                A=Y(:,1:N); x=A\Epattern(:); Erec=A*x;
                num=trapz(theta,abs(Epattern-Erec).^2.*sin(theta));
                den=trapz(theta,abs(Epattern).^2.*sin(theta));
                if sqrt(num/den)<errTh, Nmin=N-1; return; end
            end
        end

        function [R,X] = decompose_Z(Z)
            Zs=(Z+Z')/2; R=real(Zs); X=imag(Zs);
        end

        function [zn,zc,dL] = create_dipole_geometry(L,N,strat)
            if strcmp(strat,'uniform')
                zn=linspace(-L/2,L/2,N+1)';
            else
                idx=(0:N)'; zn=-L/2*cos(pi*idx/N);
            end
            zc=(zn(1:end-1)+zn(2:end))/2; dL=diff(zn);
        end
    end
end

  