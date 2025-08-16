%% main_CMA_Dipole.m (Rock-Solid Version)
% CMA Solver for Thin-Wire Dipole Antenna
% ----------------------------------------
% This script computes the Method-of-Moments impedance matrix, solves for
% characteristic modes, and analyzes their physical properties. It incorporates
% advanced features based on expert review for performance, usability, and
% scientific accuracy.
%
% USAGE:
%   results = main_CMA_Dipole('Frequency',300e6,'Length',0.48,'Radius',0.001);
%
% BENCHMARK (L=0.48λ, a=0.001λ, N=41):
%   - Mode 1 Eigenvalue (λ₁) should be ~0.
%   - Mode 1 Directivity (D₁) should be ~1.64.
%
% Author: Gemini
% Date: July 22, 2025
% Version: 19.0 (Rock-Solid, Final Expert Review)

function results_out = main_CMA_Dipole(varargin)
    %--- Parse inputs ----------------------------------------------------
    p = inputParser;
    addParameter(p,'Frequency',300e6, @(x)validateattributes(x,{'numeric'},{'vector','positive'}));
    addParameter(p,'Length',[], @(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'Radius',[], @(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'Segments',41, @(x)validateattributes(x,{'numeric'},{'scalar','integer','odd','>=',3}));
    addParameter(p,'NumModes',4, @(x)validateattributes(x,{'numeric'},{'scalar','integer','>=',1}));
    addParameter(p,'SaveOutputs',true, @islogical);
    addParameter(p,'PlotVisible',false, @islogical);
    addParameter(p,'RelTol',1e-4, @(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'AbsTol',1e-8, @(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'UseParallel',true, @islogical);
    addParameter(p,'Verbose',true, @islogical);
    parse(p,varargin{:});

    params = p.Results;
    
    canUseParallel = params.UseParallel && license('test', 'Distrib_Computing_Toolbox') && ~isempty(gcp('nocreate'));
    if params.UseParallel && ~canUseParallel && params.Verbose
        warning('Parallel pool not available or license missing. Falling back to standard for-loop.');
    end

    c   = 3e8; eta = 119.9169832*pi;
    results_out = repmat(struct(), 1, numel(params.Frequency));

    for fi = 1:numel(params.Frequency)
        f = params.Frequency(fi); lambda = c/f; k = 2*pi/lambda;
        
        if isempty(params.Length); L = 0.48*lambda; else; L = params.Length; end
        if isempty(params.Radius); a = 1e-3*lambda; else; a = params.Radius; end

        if a >= L/50
            old_a = a; a = L/100;
            warning('Radius is too large for thin-wire assumption. Clamping from %.3fmm to %.3fmm.', old_a*1e3, a*1e3);
        end

        log_msg(params.Verbose, '\n[CMA Solver] f=%.2f MHz, L=%.3fm (%.2fλ), a=%.3fmm, N=%d\n',...
                f/1e6, L, L/lambda, a*1e3, params.Segments);

        [~,~,z_center,dL] = create_dipole_geometry(L,params.Segments);
        
        log_msg(params.Verbose, 'Building impedance matrix... '); tic;
        Z = calculate_impedance_matrix(k,eta,a,z_center,dL,params.RelTol,params.AbsTol,canUseParallel);
        log_msg(params.Verbose, 'Elapsed time is %.4f seconds.\n', toc);

        [R,X] = decompose_Z(Z);
        [lambda_n,J_n] = solve_modes(X,R);

        Q_n = abs(lambda_n);
        MS_n = 1./abs(1 + 1j*lambda_n);
        [P_rad_n, D_n] = calculate_radiation_properties(J_n, k, z_center, dL);

        if params.Verbose
            fprintf('Modal Analysis Summary:\n');
            mode_indices = (1:numel(lambda_n))';
            summary_table = table(mode_indices, lambda_n, MS_n, Q_n, D_n, ...
                'VariableNames', {'Mode', 'Eigenvalue', 'ModalSignificance', 'Q_Factor_unitless', 'Directivity_unitless'});
            disp(summary_table(1:min(10, end), :));
        end

        isBenchmarkCase = (abs(L/lambda - 0.48) < 1e-3) && (abs(a/lambda - 0.001) < 1e-4);
        if isBenchmarkCase
            if abs(lambda_n(1)) > 2.0; warning('Benchmark Check: Mode 1 eigenvalue is %.3g, expected ~0.', lambda_n(1)); end
            if abs(D_n(1) - 1.64)/1.64 > 0.05; warning('Benchmark Check: Mode 1 directivity is %.3g, expected ~1.64.', D_n(1)); end
        end

        results_out(fi).frequency = f; results_out(fi).wavelength = lambda;
        results_out(fi).wavenumber = k; results_out(fi).dipole_L = L;
        results_out(fi).dipole_a = a; results_out(fi).z_center = z_center;
        results_out(fi).dL = dL;
        results_out(fi).Z_matrix = Z; results_out(fi).R_matrix = R; results_out(fi).X_matrix = X;
        results_out(fi).lambda_n = lambda_n; results_out(fi).J_n = J_n;
        results_out(fi).Q_n = Q_n; results_out(fi).MS_n = MS_n;
        results_out(fi).P_rad_n = P_rad_n; results_out(fi).Directivity_n = D_n;
        
        if params.SaveOutputs || params.PlotVisible
            plot_results(results_out(fi), params.NumModes, params.SaveOutputs, params.PlotVisible);
        end
        
        if params.SaveOutputs
            fname = sprintf('CMA_Results_%.0fMHz.mat',f/1e6);
            results = results_out(fi);
            save(fname, 'results', '-v7.3');
            log_msg(params.Verbose, 'Saved results to: %s\n', fname);
        end
    end
end

%% Main Helper Functions
function Z = calculate_impedance_matrix(k,eta,a,z_center,dL,relTol,absTol,useParallel)
    N = numel(z_center); Z = zeros(N,N,'like',1j);
    if useParallel
        parfor m = 1:N
            Z(m,:) = calculate_Z_row(m,N,z_center,k,eta,a,dL,relTol,absTol);
        end
    else
        for m = 1:N
            Z(m,:) = calculate_Z_row(m,N,z_center,k,eta,a,dL,relTol,absTol);
        end
    end
end

function Zrow = calculate_Z_row(m,N,z_center,k,eta,a,dL,relTol,absTol)
    zm = z_center(m); Zrow = zeros(1,N,'like',1j);
    const = 1j*eta/(4*pi*k);
    for n = 1:N
        z_start = z_center(n) - dL/2; z_end = z_center(n) + dL/2;
        g_integral = integral(@(zp) exp(-1j*k*Rdist(zm,zp,a))./Rdist(zm,zp,a), ...
            z_start, z_end, 'AbsTol', absTol, 'RelTol', relTol);
        term1 = k^2 * g_integral;
        zp_end = z_end; R_end = Rdist(zm, zp_end, a);
        g_end = exp(-1j*k*R_end)/R_end;
        dg_dz_end = ((zp_end - zm)/R_end) * g_end * (-1j*k - 1/R_end);
        zp_start = z_start; R_start = Rdist(zm, zp_start, a);
        g_start = exp(-1j*k*R_start)/R_start;
        dg_dz_start = ((zp_start - zm)/R_start) * g_start * (-1j*k - 1/R_start);
        term2 = dg_dz_end - dg_dz_start;
        Zrow(n) = const * (term1 + term2);
    end
end

function [P_rad, D] = calculate_radiation_properties(J_n, k, z_center, dL)
    eta = 119.9169832*pi; theta_rad = linspace(0,pi,361);
    AF_matrix = (J_n.' * exp(1j*k*(z_center.'*cos(theta_rad)))) * dL;
    U_matrix = abs(AF_matrix.*sin(theta_rad)).^2 / (2*eta);
    U_max = max(U_matrix, [], 2);
    P_rad = trapz(theta_rad, U_matrix .* 2.*pi.*sin(theta_rad), 2);
    D = (4*pi*U_max) ./ P_rad; D(P_rad < 1e-9) = 0;
end

%% Plotting Sub-system
function plot_results(results, M, savePlots, plotVisible)
    styles = get_plot_styles();
    
    plot_eigenvalues(results, M, styles, savePlots, plotVisible);
    plot_modal_significance(results, M, styles, savePlots, plotVisible);
    plot_currents(results, M, styles, savePlots, plotVisible);
    plot_patterns(results, M, styles, savePlots, plotVisible);
end

function plot_eigenvalues(r,M,s,saveFlag,plotVisible)
    f = r.frequency; lambda_n = r.lambda_n;
    if plotVisible; fig = figure('Name',sprintf('Eigenvalues @ %.0fMHz',f/1e6));
    else; fig = figure('Name',sprintf('Eigenvalues @ %.0fMHz',f/1e6),'Visible','off'); end
    plot(1:numel(lambda_n),lambda_n,'-s','Color',s.color1,'LineWidth',s.lw,'MarkerFaceColor',s.color1);
    hold on; yline(0, '--', 'Color', s.color2); hold off;
    grid on; xlabel('Mode Index'); ylabel('\lambda_n'); title('Characteristic Values');
    ylim([-20 20]);
    if saveFlag; saveas(fig,sprintf('Fig1_Eigenvalues_%.0fMHz.png',f/1e6)); end
    if ~plotVisible; close(fig); end
end

function plot_modal_significance(r,M,s,saveFlag,plotVisible)
    f = r.frequency; MS_n = r.MS_n;
    if plotVisible; fig = figure('Name',sprintf('Modal Significance @ %.0fMHz',f/1e6));
    else; fig = figure('Name',sprintf('Modal Significance @ %.0fMHz',f/1e6),'Visible','off'); end
    bar(1:numel(MS_n),MS_n,'FaceColor',s.color1);
    grid on; xlabel('Mode Index'); ylabel('Modal Significance');
    title('Modal Significance (MS = 1/|1+j\lambda_n|)');
    num_ticks = min(20, numel(MS_n));
    xticks(1:num_ticks);
    xlim([0.5 num_ticks + 0.5]); ylim([0 1.1]);
    if saveFlag; saveas(fig,sprintf('Fig2_ModalSignificance_%.0fMHz.png',f/1e6)); end
    if ~plotVisible; close(fig); end
end

function plot_currents(r,M,s,saveFlag,plotVisible)
    f = r.frequency; lambda_n = r.lambda_n; J_n = r.J_n; Q_n = r.Q_n;
    z = r.z_center; lambda = r.wavelength;
    if plotVisible; fig = figure('Name',sprintf('Eigencurrents @ %.0fMHz',f/1e6));
    else; fig = figure('Name',sprintf('Eigencurrents @ %.0fMHz',f/1e6),'Visible','off'); end
    t = tiledlayout('flow','TileSpacing','Compact');
    title(t, 'First Eigencurrents');
    for i=1:min(M,numel(lambda_n))
        ax = nexttile;
        plot(ax,z/lambda,real(J_n(:,i)),'-','Color',s.color1,'LineWidth',s.lw);
        hold(ax,'on'); plot(ax,z/lambda,imag(J_n(:,i)),'--','Color',s.color2,'LineWidth',s.lw);
        hold(ax,'off'); grid on;
        title(ax,sprintf('Mode %d: \\lambda_n=%.2f, Q_n=%.2f',i,lambda_n(i), Q_n(i)));
        xlabel(ax,'z/\lambda'); ylabel(ax,'Normalized Current');
        legend(ax,{'Re','Im'},'Location','south');
    end
    if saveFlag; saveas(fig,sprintf('Fig3_Currents_%.0fMHz.png',f/1e6)); end
    if ~plotVisible; close(fig); end
end

function plot_patterns(r,M,s,saveFlag,plotVisible)
    f = r.frequency; J_n = r.J_n; k = r.wavenumber; dL = r.dL;
    z = r.z_center; D_n = r.Directivity_n;
    if plotVisible; fig = figure('Name',sprintf('Patterns @ %.0fMHz',f/1e6));
    else; fig = figure('Name',sprintf('Patterns @ %.0fMHz',f/1e6),'Visible','off'); end
    t = tiledlayout('flow','TileSpacing','Compact');
    title(t, 'First Far-Field Patterns');
    theta = linspace(0,pi,361);
    for i=1:min(M,size(J_n,2))
        nexttile; 
        AF = (J_n(:,i).' * exp(1j*k*(z.'*cos(theta)))) * dL;
        E = abs(AF.*sin(theta)); E = E/max(E);
        polarplot(theta,E,'LineWidth',s.lw,'Color',s.color1);
        thetalim([0 180]);
        title(sprintf('Mode %d (D=%.2f)',i,D_n(i)));
    end
    if saveFlag; saveas(fig,sprintf('Fig4_Patterns_%.0fMHz.png',f/1e6)); end
    if ~plotVisible; close(fig); end
end

%% Utility Functions
function [z_start,z_end,z_center,dL] = create_dipole_geometry(L,N)
    dL = L/N; z = linspace(-L/2,L/2,N+1);
    z_start  = z(1:end-1); z_end = z(2:end); z_center = (z_start+z_end)/2;
end

function R = Rdist(z,zp,a)
    R = sqrt((z-zp).^2 + a^2);
end

function [Rmat,Xmat] = decompose_Z(Z)
    Zsym = (Z + Z.')/2; Rmat = real(Zsym); Xmat = imag(Zsym);
end

function [lambda_n,J_n] = solve_modes(X,R)
    if any(eig(R)<=0); R = R + eye(size(R))*1e-12*max(abs(diag(R))); end
    [V,D] = eig(X,R); lambda_vec = diag(D);
    [~,idx] = sort(abs(lambda_vec));
    lambda_n = lambda_vec(idx); J_n = V(:,idx);
    for i=1:size(J_n,2)
        [~,midx] = max(abs(J_n(:,i)));
        ph = angle(J_n(midx,i));
        J_n(:,i) = (J_n(:,i)*exp(-1j*ph))/max(abs(J_n(:,i)));
    end
end

function styles = get_plot_styles()
    styles.color1 = [0, 0.4470, 0.7410]; % Blue
    styles.color2 = [0.8500, 0.3250, 0.0980]; % Red
    styles.lw = 1.5; % LineWidth
end

function log_msg(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
