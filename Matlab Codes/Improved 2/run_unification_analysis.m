%% run_unification_analysis.m
% -------------------------------------------------------------------------
% This script executes a complete numerical experiment using the CmaSolver
% class to explore the relationship between internal and external antenna
% properties.
%
% It performs the following steps:
%   1. Defines a sweep over the dipole's electrical length (L/λ).
%   2. Calls the CmaSolver for each point to gather data silently.
%   3. Aggregates all simulation results.
%   4. Performs TCM analysis (counting significant modes).
%   5. Performs Far-Field complexity analysis (spherical waves).
%   6. Generates the two final "Grand Unification" plots that connect
%      the theories of TCM, Degrees of Freedom, and Radiative Limits.
%
% USAGE:
%   Ensure CmaSolver.m is in the MATLAB path, then run this script.
%
% Author: Gemini
% Date: July 27, 2025
% Version: 1.0
% -------------------------------------------------------------------------
clear; clc; close all;

%% Step 1: Define and Automate the Simulation Sweep
fprintf('--- Unification Analysis using CmaSolver ---\n');
fprintf('Step 1: Defining Simulation Sweep Parameters...\n');

% --- Constant Parameters ---
f_const = 300e6;          % Constant frequency in Hz
c = 299792458;            % Speed of light in m/s
lambda_const = c/f_const; % Reference wavelength
a = 0.001 * lambda_const; % Constant wire radius in meters
N = 41;                   % Constant number of segments

% --- Primary Sweep Variable: Electrical Length (L/λ) ---
L_over_lambda_sweep = linspace(0.1, 1.5, 51); % 51 points for a smooth curve

% --- Data Storage ---
num_points = length(L_over_lambda_sweep);
simulation_results = cell(num_points, 1);

%% --- Data Collection Loop ---
fprintf('Starting data collection sweep...\n');
tic;
for i = 1:num_points
    L_over_lambda = L_over_lambda_sweep(i);
    L = L_over_lambda * lambda_const; % Physical length for this run
    
    fprintf('Running simulation %d/%d: L/lambda = %.3f\n', i, num_points, L_over_lambda);
    
    % --- Set up parameters for the CmaSolver class ---
    params.Frequency      = f_const;
    params.Length         = L;
    params.Radius         = a;
    params.Segments       = N;
    params.BasisFunction  = 'rooftop'; % Use most accurate basis
    params.SaveOutputs    = false; % Don't save individual files
    params.PlotVisible    = false; % Don't show plots
    params.Verbose        = false; % Don't print summary tables
    params.UseParallel    = true;  % Use parallel if available

    % --- Instantiate and run the solver ---
    solver = CmaSolver(params);
    results_struct = solver.run();
    
    % Store the returned data structure
    simulation_results{i} = results_struct;
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
    physical_half_length(i) = data.dipole_L / 2;
    
    % --- Step 2: Analyze from TCM Perspective ---
    % Count modes where the eigenvalue magnitude is less than a threshold (e.g., 1)
    significant_modes = abs(data.lambda_n) < 1.0;
    NDoF_TCM(i) = sum(significant_modes);
    
    % --- Step 3: Analyze from Far-Field (Yaghjian's) Perspective ---
    % a) Calculate total current from a center-gap excitation
    NumBasisFunctions = size(data.Z_matrix, 1);
    V = zeros(NumBasisFunctions, 1);
    [~, node_idx] = min(abs(data.z_nodes - 0));
    feed_idx = max(1, node_idx - 1);
    V(feed_idx) = 1; % Delta-gap excitation at the center
    I_total = data.Z_matrix \ V;
    
    % b) Calculate the far-field pattern of this total current
    theta = linspace(0, pi, 361);
    [~, E_pattern_matrix] = CmaSolver.calculate_radiation_properties(data.VersionInfo.BasisFunction, I_total, data.wavenumber, data.z_nodes, data.z_center, data.dL, theta);
    E_pattern = E_pattern_matrix(1,:); % It returns a row vector
    if max(E_pattern) > 0; E_pattern = E_pattern / max(E_pattern); end
    
    % c) Find highest significant spherical wave mode number, N
    max_N_to_test = 20;
    fit_error_threshold = 0.05; % 5% error
    N_Yaghjian(i) = fit_spherical_waves(theta, E_pattern, max_N_to_test, fit_error_threshold);
    
    % d) Calculate the effective reactive radius 'a'
    effective_radius_a(i) = (N_Yaghjian(i) + 0.5) / data.wavenumber;
end
toc;
fprintf('Analysis complete.\n');

%% Step 4: The Grand Unification: Synthesize and Plot the Results
fprintf('Step 4: Generating final unification plots...\n');
styles = get_plot_styles(); % Get a consistent color scheme

% --- Plot 1: The DoF Unification ---
fig1 = figure('Name', 'DoF Unification', 'Position', [100, 100, 800, 600]);
plot(L_over_lambda_sweep, NDoF_TCM, '-o', 'Color', styles.color1, 'LineWidth', 2, 'MarkerFaceColor', styles.color1);
hold on;
plot(L_over_lambda_sweep, N_Yaghjian, '--s', 'Color', styles.color2, 'LineWidth', 2);
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

%% Local Helper Functions
function styles = get_plot_styles()
    styles.color1 = [0, 0.4470, 0.7410]; % Blue
    styles.color2 = [0.8500, 0.3250, 0.0980]; % Red
    styles.lw = 1.5; % LineWidth
end

function N_min = fit_spherical_waves(theta, E_pattern, max_N, error_threshold)
% This function finds the minimum number of spherical wave modes (N_min)
% required to accurately represent a given far-field pattern E_pattern.
    costh = cos(theta);
    E_pattern = E_pattern(:); % Ensure column vector
    for N = 1:max_N
        % Construct the basis matrix A, where each column is an Associated
        % Legendre Function P_n^1(cos(theta)).
        A = zeros(length(theta), N);
        for n = 1:N
            P_n_m = legendre(n, costh);
            if size(P_n_m, 1) > 1
                A(:, n) = P_n_m(2, :)'; % This is P_n^1(cos(theta))
            else
                A(:, n) = 0; % P_n^1 is zero if m>n
            end
        end
        
        % Solve the least-squares problem: A*x = E_pattern
        x = A \ E_pattern;
        % Reconstruct the pattern from the fit
        E_reconstructed = A * x;
        
        % Calculate the normalized root-mean-square error
        nrmse = norm(E_pattern - E_reconstructed) / norm(E_pattern);
        
        % Check if the fit is good enough
        if nrmse < error_threshold
            N_min = N;
            return; % Exit as soon as we find a good enough N
        end
    end
    
    % If no N meets the threshold, return the max tested value
    warning('Fit did not converge to the error threshold. Returning max_N.');
    N_min = max_N;
end
