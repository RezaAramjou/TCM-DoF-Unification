%=======================================================================
% run_phase3_unification_analysis_improved.m
%=======================================================================
% Improved Driver Script for Phase 3 Unification Analysis (v3)
% Fixes syntax errors and refines plotting
%=======================================================================

function run_phase3_unification_analysis_v3()
    %% Configuration (User-Adjustable Parameters)
    rng(0);  % Seed RNG for reproducibility
    freq_list   = [300e6];        % Frequencies to sweep (Hz)
    L_over_lambda = linspace(0.1,3.0,60); % Extended electrical length sweep
    radius_over_lambda = [0.001, 0.005, 0.01]; % Wire radius values to test
    segment_counts = [41, 81, 161];  % Mesh refinement tests

    % TCM threshold and far-field fit parameters
    tcm_threshold       = 1.0;      % |lambda_n| < threshold
    fit_error_threshold = 0.05;     % normalized RMSE
    max_spherical_modes = 30;       % max modes to test

    % Precompute Far-Field angles and Legendre basis
    theta = linspace(0,pi,361);
    costh = cos(theta);
    Pnm1 = cell(max_spherical_modes,1);
    for n = 1:max_spherical_modes
        P = legendre(n,costh); % returns [m+1 x length(theta)]
        if size(P,1) >= 2
            Pnm1{n} = P(2,:).'; % Associated Legendre P_n^1
        else
            Pnm1{n} = zeros(length(theta),1);
        end
    end

    % Initialize storage
results  = repmat(struct('L_over_lambda',[],'radius_over_lambda',[],'segments',[],'NDoF_TCM',[],'NDoF_FF',[],'a_eff',[],'fit_error',[]),0,1);
failures = repmat(struct('L_over_lambda',[],'radius',[],'segments',[],'error',[]),0,1);
timings  = [];

% Main sweep
    total_runs = numel(freq_list)*numel(radius_over_lambda)*numel(L_over_lambda)*numel(segment_counts);
    tic_total = tic;
    for f = freq_list
        c       = 3e8;
        lambda  = c/f;
        k       = 2*pi/lambda;
        for a_norm = radius_over_lambda
            a = a_norm * lambda;
            for N = segment_counts
                for Lnorm = L_over_lambda
                    start_run = tic;
                    try
                        % Call solver
                        params = struct('Frequency',f,'Length',Lnorm*lambda,'Radius',a,'Segments',N,...
                                        'SaveOutputs',false,'PlotVisible',false,'Verbose',false);
                        data = main_CMA_Dipole(params);

                        % TCM analysis
                        NDoF_TCM = sum(abs(data.lambda_n) < tcm_threshold);

                        % Far-field current and pattern
                        V       = zeros(N,1);
                        V(ceil(N/2)) = 1;
                        I_total = data.Z_matrix \ V;
                        AF      = (I_total.' * exp(1j*k*(data.z_center.'*costh))) * data.dL;
                        E_pattern = abs(AF .* sin(theta));
                        if max(E_pattern)>0
                            E_pattern = E_pattern / max(E_pattern);
                        end

                        % Fit spherical modes
                        [N_Yag, fit_err] = fit_spherical_modes(Pnm1, E_pattern, max_spherical_modes, fit_error_threshold);
                        a_eff = (N_Yag + 0.5) / k;

                        % Store result
                        res = struct();
                        res.L_over_lambda      = Lnorm;
                        res.radius_over_lambda = a_norm;
                        res.segments           = N;
                        res.NDoF_TCM           = NDoF_TCM;
                        res.NDoF_FF            = N_Yag;
                        res.a_eff              = a_eff;
                        res.fit_error          = fit_err;
                        results               = [results; res];

                        % Timing
                        timings = [timings; toc(start_run)];

                    catch ME
                        failures = [failures; struct('L_over_lambda',Lnorm,'radius',a_norm,'segments',N,'error',{ME.message})];
                    end
                end
            end
        end
    end
    total_time = toc(tic_total);

    % Save all results and metadata
    meta = struct('date',datestr(now), 'freq_list',freq_list, 'L_over_lambda',L_over_lambda,...
                 'radius_over_lambda',radius_over_lambda,'segment_counts',segment_counts,...
                 'tcm_threshold',tcm_threshold,'fit_error_threshold',fit_error_threshold,...
                 'max_spherical_modes',max_spherical_modes,'total_runs',total_runs,'total_time_s',total_time);
    save('Phase3_Unification_Results_v3.mat','results','failures','timings','meta');

    % Generate unified plots
    plot_unification(results, meta);
end

%% Helper Functions

function [N_min,err] = fit_spherical_modes(Pnm1, E_pattern, max_N, threshold)
    % Convert cell array to matrix once
    basis = cell2mat(Pnm1);
    err   = NaN;
    for N = 1:max_N
        A    = basis(:,1:N);
        x    = A \ E_pattern;         % least-squares solution
        E_rec = A * x;
        nrmse = norm(E_pattern - E_rec) / norm(E_pattern);
        if nrmse < threshold
            N_min = N;
            err   = nrmse;
            return;
        end
    end
    N_min = max_N;
    err   = nrmse;
end

function plot_unification(results, meta)
    % Select representative case: first radius, finest mesh
    r      = meta.radius_over_lambda(1);
    Nmesh  = meta.segment_counts(end);
    mask   = [results.radius_over_lambda]==r & [results.segments]==Nmesh;
    sel    = results(mask);
    L      = [sel.L_over_lambda];
    N_TCM  = [sel.NDoF_TCM];
    N_FF   = [sel.NDoF_FF];
    a_eff  = [sel.a_eff];

    % Theoretical DoF curve: approx 2L/λ
    N_th = 2 * L;

    styles = get_plot_styles();
    figure; hold on; grid on; box on;
    plot(L, N_TCM, '-o', 'Color', styles.color1, 'LineWidth', 2, 'MarkerFaceColor', styles.color1);
    plot(L, N_FF , '--s','Color', styles.color2, 'LineWidth', 2);
    plot(L, N_th , '-.','Color', styles.color3, 'LineWidth', 1.5);
    xlabel('L/\lambda'); ylabel('Number of DoF');
    title('Unification of Degrees of Freedom');
    legend('TCM (mode count)','Far-Field Fit','Theory (~2L/\lambda)','Location','northwest');
    set(gca, 'FontSize', 12, 'FontWeight', 'bold');

    % Physical half-length curve
    f     = meta.freq_list(1);
    c     = 3e8;
    lambda = c/f;
    physical = (L * lambda) / 2;  % meters

    figure; hold on; grid on; box on;
    plot(L, a_eff, '-d', 'Color', styles.color1, 'LineWidth', 2, 'MarkerFaceColor', styles.color1);
    plot(L, physical, '--^', 'Color', styles.color2, 'LineWidth', 2);
    xlabel('L/\lambda'); ylabel('Radius (m)');
    title('Effective Reactive vs. Physical Half-Length');
    legend('Effective Radius a_{eff}', 'Physical Half-Length L/2','Location','northwest');
    set(gca, 'FontSize', 12, 'FontWeight', 'bold');
end

function styles = get_plot_styles()
    styles.color1 = [0,0.4470,0.7410];
    styles.color2 = [0.8500,0.3250,0.0980];
    styles.color3 = [0.4660,0.6740,0.1880];
end
