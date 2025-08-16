%% run_cma_analysis.m
% -------------------------------------------------------------------------
% This script demonstrates the full capabilities of CmaSolver v4.0.
% It showcases higher-order basis functions, adaptive meshing, and the
% comprehensive benchmark suite with far-field pattern validation.
%
% USAGE:
%   Ensure CmaSolver.m and TestCmaSolver.m are in the same directory or
%   on the MATLAB path, then run this script.
%
% Author: Gemini
% Date: July 27, 2025
% Version: 4.0 (Advanced Numerics & Testing)
% -------------------------------------------------------------------------

clear; close all; clc;

% --- Advanced Analysis: Higher-Order Basis & Adaptive Mesh ---
fprintf('--- Running Advanced Analysis with Rooftop Basis and Center-Biased Meshing ---\n');
params.Frequency      = 300e6;
params.Length         = 0.48; % meters
params.Radius         = 0.001; % meters
params.Segments       = 51;
params.NumModes       = 4;
params.PlotVisible    = true;
params.SaveOutputs    = false; % Disable saving for this demo

% Use the new advanced features
params.BasisFunction    = 'rooftop'; % 'pulse' or 'rooftop'
params.MeshingStrategy  = 'center-biased'; % 'uniform' or 'center-biased'

solver = CmaSolver(params);
results = solver.run();
fprintf('Advanced analysis complete.\n\n');
disp('Note the new VersionInfo struct in the results:');
disp(results.VersionInfo);


% --- Rigorous Benchmark Analysis with Pattern Validation ---
fprintf('\n--- Running Rigorous Benchmark Analysis (Impedance and Pattern) ---\n');

% Create a more realistic "experimental" reference file with slight noise
% and a reference far-field pattern.
fprintf('Creating a dummy experimental reference file: experimental_data.mat\n');
ref_freq = linspace(250e6, 350e6, 21);
% Plausible impedance curve with some noise
ref_R = 10 + 60 * exp(-((ref_freq - 300e6)/30e6).^2) + 2*randn(size(ref_freq));
ref_X = 150 * (ref_freq - 290e6) / 50e6 + 3*randn(size(ref_freq));
ref_Z_in = ref_R + 1j * ref_X;
% Plausible far-field pattern for a dipole (sin(theta))
ref_theta = linspace(0, pi, 181);
ref_pattern = sin(ref_theta);
save('experimental_data.mat', 'ref_freq', 'ref_Z_in', 'ref_theta', 'ref_pattern');

bench_params = params;
bench_params.PlotVisible = true;
bench_params.Benchmark.Enabled = true;
bench_params.Benchmark.ReferenceFile = 'experimental_data.mat';
bench_params.Benchmark.ModeToCompare = 1;

bench_solver = CmaSolver(bench_params);
benchmark_results = bench_solver.runBenchmark();
fprintf('Benchmark analysis complete.\n\n');


% --- Run the Unit Tests ---
% This demonstrates how to use the new TestCmaSolver class to verify the
% solver's integrity.
fprintf('\n--- Running Unit Test Suite ---\n');
try
    test_results = runtests('TestCmaSolver');
    disp(test_results);
catch ME
    warning('Could not run unit tests. Ensure TestCmaSolver.m is in the path. Error: %s', ME.message);
end
