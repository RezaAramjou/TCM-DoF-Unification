%% main_CMA_Dipole.m (Rock-Solid Complete Version)
% CMA Solver for Thin-Wire Dipole Antenna
%----------------------------------------
% Computes MoM impedance, solves characteristic modes, evaluates radiation,
% and saves eigenvalue/directivity plots when requested.
% Author: Gemini
% Date: July 22, 2025 (Complete)

function results_out = main_CMA_Dipole(varargin)
    %% Parse inputs
    p = inputParser;
    addParameter(p,'Frequency',300e6,@(x)validateattributes(x,{'numeric'},{'vector','positive'}));
    addParameter(p,'Length',[],@(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'Radius',[],@(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'Segments',41,@(x)validateattributes(x,{'numeric'},{'scalar','integer','odd','>=',3}));
    addParameter(p,'NumModes',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','>=',1}));
    addParameter(p,'SaveOutputs',false,@islogical);
    addParameter(p,'PlotVisible',false,@islogical);
    addParameter(p,'RelTol',1e-4,@(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'AbsTol',1e-8,@(x)validateattributes(x,{'numeric'},{'scalar','positive'}));
    addParameter(p,'UseParallel',false,@islogical);
    addParameter(p,'Verbose',false,@islogical);
    parse(p,varargin{:});
    params = p.Results;

    %% Constants
    c   = 3e8;
    eta = 119.9169832*pi;

        %% Prepare output
    Nf = numel(params.Frequency);
    template = struct( ...
    'frequency',     [], ...
    'lambda_n',      [], ...
    'J_n',           [], ...
    'P_rad_n',       [], ...
    'Directivity_n', [], ...
    'Z_matrix',      [], ...
    'z_center',      [], ...
    'dL',            []  ...
);
    results_out = repmat(template, 1, Nf);

    for fi = 1:Nf
        f      = params.Frequency(fi);
        lambda = c/f;
        k      = 2*pi/lambda;

        %% Dipole dimensions
        if isempty(params.Length)
            L = 0.48*lambda;
        else
            L = params.Length;
        end
        if isempty(params.Radius)
            a = 1e-3*lambda;
        else
            a = params.Radius;
        end
        % Thin-wire assumption clamp
        if a >= L/50
            a = L/100;
            warning('Clamped radius to satisfy thin-wire assumption.');
        end

        %% Geometry discretization
        [z_start,z_end,z_center,dL] = create_dipole_geometry(L,params.Segments);

        %% Impedance matrix via MoM
        Z = calculate_impedance_matrix(k,eta,a,z_center,dL,params.RelTol,params.AbsTol,params.UseParallel);

        %% Modal decomposition (TCM)
        [Rmat,Xmat] = decompose_Z(Z);
        [lambda_n,J_n] = solve_modes(Xmat,Rmat,params.NumModes);

        %% Radiation properties
        [P_rad_n,D_n] = calculate_radiation_properties(J_n,k,z_center,dL);

        %% Store results
        res = struct();
        res.frequency     = f;
        res.lambda_n      = lambda_n;
        res.J_n           = J_n;
        res.P_rad_n       = P_rad_n;
        res.Directivity_n = D_n;
        res.Z_matrix      = Z;
        res.z_center      = z_center;
        res.dL            = dL;
        results_out(fi)   = res;

        %% Plot and save if requested
        plot_results(res,params.NumModes,params.SaveOutputs,params.PlotVisible);
    end
end

%% Geometry helper
function [z_start,z_end,z_center,dL] = create_dipole_geometry(L,N)
    dL = L/N;
    z = linspace(-L/2,L/2,N+1);
    z_start  = z(1:end-1);
    z_end    = z(2:end);
    z_center = (z_start + z_end)/2;
end

%% Impedance matrix assembly
function Z = calculate_impedance_matrix(k,eta,a,z_center,dL,relTol,absTol,usePar)
    N = numel(z_center);
    Z = complex(zeros(N,N));
    if usePar
        parfor m = 1:N
            Z(m,:) = impedance_row(m,N,z_center,k,eta,a,dL,relTol,absTol);
        end
    else
        for m = 1:N
            Z(m,:) = impedance_row(m,N,z_center,k,eta,a,dL,relTol,absTol);
        end
    end
end

%% Single row of Z
function Zrow = impedance_row(m,N,z_center,k,eta,a,dL,relTol,absTol)
    zm = z_center(m);
    Zrow = complex(zeros(1,N));
    coeff = 1j*eta/(4*pi*k);
    for n = 1:N
        z1 = z_center(n) - dL/2;
        z2 = z_center(n) + dL/2;
        % Green's function integral term
        Gint = integral(@(zp) exp(-1j*k*sqrt((zm-zp).^2 + a^2))./sqrt((zm-zp).^2 + a^2),...
                        z1, z2, 'RelTol',relTol, 'AbsTol',absTol);
        % derivative of G at endpoints
        dG2 = dGreen(zm,z2,k,a);
        dG1 = dGreen(zm,z1,k,a);
        Zrow(n) = coeff*(k^2*Gint + dG2 - dG1);
    end
end

%% Green's function derivative
function val = dGreen(zm,zp,k,a)
    R = sqrt((zm-zp).^2 + a^2);
    G = exp(-1j*k*R)./R;
    val = ((zp - zm)/R).*( -1j*k - 1/R ).*G;
end

%% Decompose Z into R and X
function [Rmat,Xmat] = decompose_Z(Z)
    Zs = (Z + Z.')/2;
    Rmat = real(Zs);
    Xmat = imag(Zs);
end

%% Solve generalized eigenproblem X v = lambda R v
function [lambda_n,J_n] = solve_modes(X,R,M)
    % Regularize R if needed
    if any(eig(R) <= 0)
        R = R + eye(size(R))*1e-12*max(abs(diag(R)));
    end
    [V,D] = eig(X,R);
    lam = diag(D);
    [~,idx] = sort(abs(lam),'ascend');
    idx = idx(1:M);
    lambda_n = lam(idx);
    J_n = V(:,idx);
    % Phase normalization
    for i = 1:M
        [~,i0] = max(abs(J_n(:,i)));
        ph = angle(J_n(i0,i));
        J_n(:,i) = J_n(:,i)*exp(-1j*ph)/max(abs(J_n(:,i)));
    end
end

%% Radiation properties: P_rad and directivity
function [P_rad,D] = calculate_radiation_properties(J_n,k,z_center,dL)
    eta = 119.9169832*pi;
    theta = linspace(0,pi,361);
    M = size(J_n,2);
    P_rad = zeros(M,1);
    D     = zeros(M,1);
    for i = 1:M
        AF = (J_n(:,i).' * exp(1j*k*(z_center.'*cos(theta)))) * dL;
        U  = abs(AF .* sin(theta)).^2 / (2*eta);
        P_rad(i) = trapz(theta, U * 2*pi .* sin(theta));
        D(i)     = 4*pi * max(U) / P_rad(i);
    end
end

%% Plotting function: eigenvalues and directivity
function plot_results(res,M,saveFlag,plotVis)
    styles = get_plot_styles();
    f      = res.frequency;
    lam    = res.lambda_n;
    D      = res.Directivity_n;

    % Eigenvalues plot
    fig1 = figure('Visible', ternary(plotVis,'on','off'));
    plot(1:numel(lam), lam, '-o', 'Color',styles.color1, 'LineWidth',2, 'MarkerFaceColor',styles.color1);
    hold on; yline(0,'--','Color',styles.color2); hold off;
    grid on;
    xlabel('Mode Index'); ylabel('\lambda_n');
    title(sprintf('Eigenvalues @ %.0f MHz', f/1e6));
    if saveFlag
        saveas(fig1, sprintf('Eigenvalues_%.0fMHz.png', f/1e6));
    end
    if ~plotVis, close(fig1); end

    % Directivity bar chart
    fig2 = figure('Visible', ternary(plotVis,'on','off'));
    bar(1:min(M,numel(D)), D(1:min(M,numel(D))), 'FaceColor',styles.color1);
    grid on;
    xlabel('Mode Index'); ylabel('Directivity');
    title(sprintf('Directivity @ %.0f MHz', f/1e6));
    if saveFlag
        saveas(fig2, sprintf('Directivity_%.0fMHz.png', f/1e6));
    end
    if ~plotVis, close(fig2); end
end

%% Ternary helper
function out = ternary(cond,a,b)
    if cond, out = a; else out = b; end
end

%% Plot styles
function s = get_plot_styles()
    s.color1 = [0,0.4470,0.7410];
    s.color2 = [0.8500,0.3250,0.0980];
end
