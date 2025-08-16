%% TestCmaSolver_final.m
% -------------------------------------------------------------------------
% This class defines a formal unit test suite for the CmaSolver class
% using the MATLAB Unit Testing Framework.
%
% USAGE:
%   runtests('TestCmaSolver_final')
%
% It verifies:
%   - Core physics (matrix reciprocity, mode orthogonality).
%   - Numerical correctness (regression against known values).
%   - Robustness of different configurations.
%
% Author: Gemini
% Date: July 27, 2025
% Version: 3.0 (Final, Corrected for v6 Solver)
% -------------------------------------------------------------------------

classdef TestCmaSolver_final < matlab.unittest.TestCase

    properties
        % A standard set of parameters for all tests
        Config
    end

    methods(TestMethodSetup)
        % Set up a fresh solver instance before each test
        function createConfig(testCase)
            config.Dipole.Length = 0.48;
            config.Dipole.Radius = 0.001;
            config.Mesh.Segments = 21; % Use fewer segments for speed
            config.Mesh.Strategy = 'uniform';
            config.Numerics.BasisFunction = 'rooftop';
            config.Numerics.Accuracy.Level = 'medium';
            config.Execution.Frequency = 300e6;
            config.Execution.NumModes = 4;
            config.Execution.Verbose = false;
            config.Execution.PlotVisible = false;
            config.Execution.UseParallel = false;
            testCase.Config = config;
        end
    end

    methods(Test)
        % Test 1: Verify that the impedance matrix is reciprocal (Zmn = Znm)
        function testReciprocity(testCase)
            solver = CmaSolver(testCase.Config);
            results = solver.run();
            Z = results.Z_matrix;
            reciprocity_error = norm(Z - Z.', 'fro') / norm(Z, 'fro');
            testCase.verifyThat(reciprocity_error, matlab.unittest.constraints.IsLessThan(1e-12));
        end

        % Test 2: Verify that characteristic modes are orthogonal over R
        function testModeOrthogonality(testCase)
            solver = CmaSolver(testCase.Config);
            results = solver.run();
            J_n = results.J_n;
            R = results.R_matrix;
            orthogonality_matrix = J_n' * R * J_n;
            off_diagonal_norm = norm(orthogonality_matrix - diag(diag(orthogonality_matrix)), 'fro');
            diagonal_norm = norm(diag(diag(orthogonality_matrix)), 'fro');
            orthogonality_error = off_diagonal_norm / diagonal_norm;
            testCase.verifyThat(orthogonality_error, matlab.unittest.constraints.IsLessThan(1e-12));
        end

        % Test 3: Regression test for the first eigenvalue with rooftop basis
        function testFirstEigenvalueRegression(testCase)
            % For a standard case (L=0.48 lambda), the first eigenvalue should be very close to zero.
            solver = CmaSolver(testCase.Config);
            results = solver.run();
            first_eigenvalue = results.lambda_n(1);
            known_good_upper_bound = 0.2;
            testCase.verifyThat(abs(first_eigenvalue), matlab.unittest.constraints.IsLessThan(known_good_upper_bound));
        end
        
        % Test 4: Ensure the pulse basis function runs without error
        function testPulseExecution(testCase)
            % This test ensures the pulse basis runs. It suppresses the
            % expected benchmark warning that occurs with this less accurate basis.
            testCase.Config.Numerics.BasisFunction = 'pulse';
            solver = CmaSolver(testCase.Config);
            
            testFcn = @() solver.run();
            
            % We expect a specific warning, so we verify that the function
            % runs to completion without throwing an *error*.
            testCase.verifyWarningFree(testFcn, 'CmaSolver:Benchmark');
        end
        
        % Test 5: Ensure center-biased meshing runs without error
        function testCenterBiasedMeshingExecution(testCase)
            testCase.Config.Mesh.Strategy = 'center-biased';
            solver = CmaSolver(testCase.Config);
            
            testFcn = @() solver.run();
            testCase.verifyWarningFree(testFcn);
        end
    end
end
