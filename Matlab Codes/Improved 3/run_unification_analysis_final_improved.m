%% run_unification_analysis_final_improved.m
  clear; clc; close all;
  fprintf('--- Phase 3: Unification Analysis (v7.1, Serial) ---\n');
  f0=300e6; lambda=299792458/f0; a=lambda/2000;
  Llam=linspace(0.1,1.5,51); Npts=numel(Llam);
  TCM_DOF=zeros(Npts,1); Yag_DOF=zeros(Npts,1); PhysHalf=zeros(Npts,1);
  for i=1:Npts
      L=Llam(i)*lambda; PhysHalf(i)=L/2;
      meshPts=2*round(20*Llam(i))+1;
      cfg=struct(); cfg.Dipole.Length=L; cfg.Dipole.Radius=a;
      cfg.Mesh.Segments=meshPts; cfg.Convergence.Step=2;
      solver=CmaSolver(cfg); res=solver.run();
      lambda_n=res(1).lambda_n; TCM_DOF(i)=sum(abs(lambda_n)<1);
      Z=res(1).Z_matrix; V=zeros(size(Z,1),1); V(1)=1; I=Z\V;
      theta=linspace(0,pi,181);
      [~,Epat]=CmaSolver.calculate_radiation_properties([],I,res(1).wavenumber,[],[],[],theta);
      Yag_DOF(i)=CmaSolver.fit_spherical_waves(theta,Epat,20,0.05);
  end
  figure; plot(Llam,TCM_DOF,'-o',Llam,Yag_DOF,'-s'); grid on;
  xlabel('L/\lambda'); ylabel('DoF'); legend('TCM','Far-Field');
  figure; plot(Llam,PhysHalf,'-^',Llam,(Yag_DOF+0.5)/(2*pi/lambda),'-d'); grid on;
  xlabel('L/\lambda'); ylabel('Radius (m)'); legend('L/2','a_{eff}');
  fprintf('Done.\n');
