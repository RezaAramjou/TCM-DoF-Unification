%% run_unification_analysis_final.m
% -------------------------------------------------------------------------
% This script executes a complete numerical experiment using the CmaSolver
% class to explore the relationship between internal and external antenna
% properties. This version runs in serial mode only.
%
% It performs the following steps:
%   1. Defines a sweep over the dipole's electrical length (L/λ).
%   2. Calls the CmaSolver efficiently for each point to gather data.
%   3. Aggregates all simulation results.
%   4. Performs TCM analysis (counting significant modes).
%   5. Performs a robust Far-Field complexity analysis (spherical waves).
%   6. Generates the two final "Grand Unification" plots.
%
% Author: Gemini
% Date: July 28, 2025
% Version: 6.1 (Serial Execution)
% -------------------------------------------------------------------------
clear; clc; close all;

%% Step 1: Define and Automate the Simulation Sweep
fprintf('--- Phase 3: Unification Analysis (v6.1 Solver) ---\n');
fprintf('Step 1: Defining Simulation Sweep Parameters...\n');

% --- Constant Parameters ---
f_const = 300e6;
c = 299792458;
lambda_const = c/f_const;
a_const = 0.001 * lambda_const;
N_const = 41;

% --- Primary Sweep Variable: Electrical Length (L/λ) ---
L_over_lambda_sweep = linspace(0.1, 1.5, 51);

% --- Data Storage ---
num_points = length(L_over_lambda_sweep);
simulation_results = cell(num_points, 1);

%% --- Data Collection Loop ---
fprintf('Starting data collection sweep...\n');
tic;
for i = 1:num_points
    L_over_lambda = L_over_lambda_sweep(i);
    L = L_over_lambda * lambda_const;
    
    fprintf('Running simulation %d/%d: L/lambda = %.3f\n', i, num_points, L_over_lambda);
    
    % --- Configure the solver for an efficient "silent" run ---
    config.Dipole.Length = L;
    config.Dipole.Radius = a_const;
    config.Mesh.Segments = N_const;
    config.Mesh.Strategy = 'uniform';
    config.Numerics.BasisFunction = 'rooftop';
    config.Numerics.Accuracy.Level = 'medium';
    config.Execution.Frequency = f_const;
    config.Execution.NumModes = 20; % Calculate enough modes for analysis
    config.Execution.Verbose = false;     % Disable console tables
    config.Execution.PlotVisible = false; % Disable figure generation
    config.Execution.StoreZMatrix = true; % Keep Z-matrix for post-processing

    solver = CmaSolver(config);
    simulation_results{i} = solver.run();
end
toc;
fprintf('Data collection sweep complete.\n');

%% Step 2 & 3: Analyze the Results
fprintf('Steps 2 & 3: Performing TCM and Far-Field Analysis...\n');
tic;

% --- Initialize result vectors ---
NDoF_TCM = zeros(num_points, 1);
N_Yaghjian = zeros(num_points, 1);
effective_radius_a = zeros(num_points, 1);
physical_half_length = zeros(num_points, 1);

for i = 1:num_points
    data = simulation_results{i};
    physical_half_length(i) = data.Dipole.Length / 2;
    
    % --- Step 2: Analyze from TCM Perspective ---
    significant_modes = abs(data.lambda_n) < 1.0;
    NDoF_TCM(i) = sum(significant_modes);
    
    % --- Step 3: Analyze from Far-Field (Yaghjian's) Perspective ---
    NumBasisFunctions = size(data.Z_matrix, 1);
    V = zeros(NumBasisFunctions, 1);
    [~, node_idx] = min(abs(data.z_nodes - 0));
    feed_idx = max(1, node_idx - 1);
    V(feed_idx) = 1;
    I_total = data.Z_matrix \ V;
    
    theta = linspace(0, pi, 361);
    [~, E_pattern_matrix] = CmaSolver.calculate_radiation_properties(data.VersionInfo.BasisFunction, I_total, data.wavenumber, data.z_nodes, data.z_center, data.dL, theta);
    E_pattern = E_pattern_matrix(1,:);
    if max(E_pattern) > 0; E_pattern = E_pattern / max(E_pattern); end
    
    max_N_to_test = 20;
    fit_error_threshold = 0.05;
    N_Yaghjian(i) = fit_spherical_waves(theta, E_pattern, max_N_to_test, fit_error_threshold);
    
    effective_radius_a(i) = (N_Yaghjian(i) + 0.5) / data.wavenumber;
end
toc;
fprintf('Analysis complete.\n');

%% Step 4: The Grand Unification: Synthesize and Plot the Results
fprintf('Step 4: Generating final unification plots...\n');
styles = CmaSolver.get_plot_styles();

% --- Plot 1: The DoF Unification ---
fig1 = figure('Name', 'DoF Unification', 'Position', [100, 100, 800, 600]);
plot(L_over_lambda_sweep, NDoF_TCM, '-o', 'Color', styles.Color1, 'LineWidth', 2, 'MarkerFaceColor', styles.Color1);
hold on;
plot(L_over_lambda_sweep, N_Yaghjian, '--s', 'Color', styles.Color2, 'LineWidth', 2);
grid on; box on;
xlabel('Dipole Length (L/\lambda)');
ylabel('Number of Degrees of Freedom (NDoF)');
title('Unification of DoF: Internal Modes vs. External Field Complexity');
legend('NDoF from TCM (|λ_n| < 1)', 'NDoF from Far-Field (N_{Yaghjian})', 'Location', 'northwest');
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
saveas(fig1, 'Fig_Unification_DoF.png');

% --- Plot 2: Effective Size vs. Physical Size ---
fig2 = figure('Name', 'Effective Size vs. Physical Size', 'Position', [950, 100, 800, 600]);
plot(L_over_lambda_sweep, effective_radius_a, '-d', 'Color', [0.1 0.7 0.2], 'LineWidth', 2, 'MarkerFaceColor', [0.1 0.7 0.2]);
hold on;
plot(L_over_lambda_sweep, physical_half_length, '--^', 'Color', [0.6 0.2 0.8], 'LineWidth', 2);
grid on; box on;
xlabel('Dipole Length (L/\lambda)');
ylabel('Radius (meters)');
title('Effective Radiating Size vs. Physical Size');
legend('Effective Reactive Radius a_{eff}', 'Physical Half-Length L/2', 'Location', 'northwest');
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
saveas(fig2, 'Fig_Unification_Size.png');

fprintf('Plots saved. Unification analysis is complete.\n');

%% Local Helper Function for Far-Field Analysis
function N_min = fit_spherical_waves(theta, E_pattern, max_N, error_threshold)
    costh = cos(theta);
    E_pattern = E_pattern(:);
    for N = 1:max_N
        A = zeros(length(theta), N);
        for n = 1:N
            P_n_m = legendre(n, costh);
            if size(P_n_m, 1) > 1; A(:, n) = P_n_m(2, :)'; end
        end
        
        if cond(A) > 1e10
            warning('fit_spherical_waves:IllConditioned', 'Fit matrix is ill-conditioned (cond=%.2e). Using pseudoinverse.', cond(A));
            x = pinv(A) * E_pattern;
        else
            x = A \ E_pattern;
        end
        
        E_reconstructed = A * x;
        nrmse = norm(E_pattern - E_reconstructed) / norm(E_pattern);
        
        if nrmse < error_threshold
            N_min = N;
            return;
        end
    end
    
    warning('fit_spherical_waves:NoConvergence', 'Fit did not converge to the error threshold. Returning max_N.');
    N_min = max_N;
end
