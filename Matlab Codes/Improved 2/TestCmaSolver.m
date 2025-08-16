%% TestCmaSolver.m
% -------------------------------------------------------------------------
% This class defines a formal unit test suite for the CmaSolver class
% using the MATLAB Unit Testing Framework.
%
% USAGE:
%   runtests('TestCmaSolver')
%
% It verifies:
%   - Core physics (matrix reciprocity, mode orthogonality).
%   - Numerical correctness (regression against known values).
%   - Robustness of different configurations.
%
% Author: Gemini
% Date: July 27, 2025
% Version: 2.2 (Final, Corrected for All Solver Versions)
% -------------------------------------------------------------------------

classdef TestCmaSolver < matlab.unittest.TestCase

    properties
        % A standard set of parameters for all tests
        TestParams
        Solver
    end

    methods(TestMethodSetup)
        % Set up a fresh solver instance before each test
        function createSolver(testCase)
            params.Frequency      = 300e6;
            params.Length         = 0.48;
            params.Radius         = 0.001;
            params.Segments       = 21; % Use fewer segments for speed
            params.Verbose        = false;
            params.PlotVisible    = false;
            params.UseParallel    = false; % Ensure deterministic serial execution for tests
            testCase.TestParams = params;
            testCase.Solver = CmaSolver(params);
        end
    end

    methods(Test)
        % Test 1: Verify that the impedance matrix is reciprocal (Zmn = Znm)
        function testReciprocity(testCase)
            testCase.Solver.BasisFunction = 'rooftop'; % Use the most accurate basis
            results = testCase.Solver.run();
            Z = results.Z_matrix;
            
            reciprocity_error = norm(Z - Z.', 'fro') / norm(Z, 'fro');
            
            testCase.verifyThat(reciprocity_error, ...
                matlab.unittest.constraints.IsLessThan(1e-12), ...
                'The impedance matrix Z must be reciprocal (Z=Z^T).');
        end

        % Test 2: Verify that characteristic modes are orthogonal over R
        function testModeOrthogonality(testCase)
            testCase.Solver.BasisFunction = 'rooftop'; % Use the most accurate basis
            results = testCase.Solver.run();
            J_n = results.J_n;
            R = results.R_matrix;
            
            orthogonality_matrix = J_n' * R * J_n;
            
            off_diagonal_norm = norm(orthogonality_matrix - diag(diag(orthogonality_matrix)), 'fro');
            diagonal_norm = norm(diag(diag(orthogonality_matrix)), 'fro');
            
            orthogonality_error = off_diagonal_norm / diagonal_norm;
            
            testCase.verifyThat(orthogonality_error, ...
                matlab.unittest.constraints.IsLessThan(1e-12), ...
                'Characteristic currents J_n must be orthogonal with respect to the R matrix.');
        end

        % Test 3: Regression test for the first eigenvalue
        function testFirstEigenvalueRegression(testCase)
            % For a standard case (L=0.48 lambda), the first eigenvalue should be very close to zero.
            % This is a fundamental benchmark. Use the most accurate method.
            testCase.Solver.BasisFunction = 'rooftop';
            results = testCase.Solver.run();
            first_eigenvalue = results.lambda_n(1);
            
            % A known-good result for this configuration is typically < 0.2
            known_good_upper_bound = 0.2;
            
            testCase.verifyThat(abs(first_eigenvalue), ...
                matlab.unittest.constraints.IsLessThan(known_good_upper_bound), ...
                'The first eigenvalue for a resonant dipole is significantly different from the expected near-zero value.');
        end
        
        % Test 4: Ensure the pulse basis function runs without error
        function testPulseExecution(testCase)
            % This is a smoke test to ensure the pulse implementation
            % completes a run without numerical errors. A warning for the
            % benchmark check is expected for this less accurate basis, so
            % we do not use verifyWarningFree here.
            testCase.Solver.BasisFunction = 'pulse';
            
            try
                testCase.Solver.run();
                testCase.verifyTrue(true); % If it reaches here, no error was thrown
            catch ME
                testCase.verifyFail(['Running with pulse basis threw an unexpected error: ' ME.message]);
            end
        end
        
        % Test 5: Ensure center-biased meshing runs without error
        function testCenterBiasedMeshingExecution(testCase)
            testCase.Solver.MeshingStrategy = 'center-biased';
            testCase.Solver.BasisFunction = 'rooftop'; % Use accurate basis to avoid benchmark warnings
            
            testFcn = @() testCase.Solver.run();
            testCase.verifyWarningFree(testFcn, ...
                'Running the solver with center-biased meshing should not produce warnings.');
        end
    end
end
