%% run_analysis_final.m
% -------------------------------------------------------------------------
% This script demonstrates the full capabilities of the rock-solid CmaSolver
% v6.0. It showcases the new, cleaner configuration API and the robust
% analysis features.
%
% USAGE:
%   Ensure CmaSolver.m and TestCmaSolver.m are in the same directory or
%   on the MATLAB path, then run this script.
%
% Author: Gemini
% Date: July 27, 2025
% Version: 6.0 (Final, Post-Review Refactor)
% -------------------------------------------------------------------------

clear; close all; clc;

% --- Step 1: Define the Configuration ---
% The new API uses a single, hierarchical config struct.

% 1a. Physical Properties of the Antenna
config.Dipole.Length = 0.48;  % meters
config.Dipole.Radius = 0.001; % meters

% 1b. Discretization/Meshing Properties
config.Mesh.Segments = 51;
config.Mesh.Strategy = 'center-biased'; % 'uniform' or 'center-biased'

% 1c. Numerical Method Properties
config.Numerics.BasisFunction = 'rooftop'; % 'pulse' or 'rooftop'
config.Numerics.Accuracy.Level = 'high'; % 'low', 'medium', or 'high'

% 1d. Execution and Output Properties
config.Execution.Frequency = 300e6;
config.Execution.NumModes = 4;
config.Execution.UseParallel = true;
config.Execution.Verbose = true;
config.Execution.PlotVisible = true;
config.Execution.StoreZMatrix = true; % Keep Z-matrix for this single run

% --- Step 2: Run a Standard Analysis ---
fprintf('--- Running Standard Single-Frequency Analysis ---\n');
solver = CmaSolver(config);
results = solver.run();
fprintf('Standard analysis complete.\n\n');


% --- Step 3: Run a Rigorous Benchmark ---
fprintf('\n--- Running Rigorous Benchmark Analysis ---\n');

% Create a reference file with version info
fprintf('Creating a dummy benchmark reference file: benchmark_data_v6.mat\n');
ref_freq = linspace(250e6, 350e6, 11);
ref_R = 10 + 60 * exp(-((ref_freq - 300e6)/30e6).^2);
ref_X = 150 * (ref_freq - 290e6) / 50e6;
ref_Z_in = ref_R + 1j * ref_X;
ref_theta = linspace(0, pi, 181);
ref_pattern = sin(ref_theta);
VersionInfo.SolverVersion = "6.0"; % Match the solver version
VersionInfo.BasisFunction = "rooftop";
save('benchmark_data_v6.mat', 'ref_freq', 'ref_Z_in', 'ref_theta', 'ref_pattern', 'VersionInfo');

% Configure the benchmark run
bench_config = config;
bench_config.Execution.PlotVisible = true;
bench_config.Benchmark.Enabled = true;
bench_config.Benchmark.ReferenceFile = 'benchmark_data_v6.mat';
bench_config.Benchmark.ModeToCompare = 1;
bench_config.Benchmark.ForceSerial = true; % Ensure reproducible results

bench_solver = CmaSolver(bench_config);
benchmark_results = bench_solver.runBenchmark();
fprintf('Benchmark analysis complete.\n\n');


% --- Step 4: Run the Unit Tests ---
fprintf('\n--- Running Unit Test Suite ---\n');
try
    test_results = runtests('TestCmaSolver_final');
    disp(test_results);
catch ME
    warning('Could not run unit tests. Ensure TestCmaSolver_final.m is in the path. Error: %s', ME.message);
end
