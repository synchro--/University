%% Basic usage of Tensorlab demo
%    Tensorlab is a Matlab package for complex optimization and tensor
%    computations. This demo will discuss the basics of Tensorlab. It
%    consists of three consecutive parts. A first section will explain how
%    a tensor can be defined and visualized. The second section handles the
%    elementary use of the ``cpd`` command, while the third section
%    discusses more advanced items such as customized initialization
%    methods and algorithms.
%
%    A more detailed discussion can be found <a href="matlab:
%    web('http://www.tensorlab.net/demos/basic.html', '-browser')">online</a>.
%
%    See also cpd, cpderr, cpd_nls, mlsvd, lmlra

% Authors: Otto Debals         (Otto.Debals@esat.kuleuven.be)
%          Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
%
% Version History:
% - 2016/03/26   NV      Added visualize
% - 2015/07/12   OD      Initial version

clear variables; close all; clc
rng('default'); rng(50)

fprintf(['Tensorlab is a Matlab package for complex optimization and tensor\n' ...
    'computations. This demo will discuss the basics of Tensorlab. It\n' ...
    'consists of three consecutive parts. A first section ''Tensor\n' ...
    'construction and visualization'' will explain how a tensor can be\n' ...
    'defined and visualized. The second section ''Canonical polyadic\n' ...
    'decomposition for beginners'' handles the elementary use of the\n' ...
    'cpd command, while the third section ''CPD for pros'' discusses more\n' ...
    'advanced items such as customized initialization methods and algorithms.\n']);
fprintf('An accompanying text can be found %s.\n\n',gendemolink('basic','online'));
fprintf(['Go to <a href="matlab: web(''http://www.tensorlab.net'', ''-browser'')">www.tensorlab.net</a> for Tensorlab\''s documentation and further demos.\n' ...
    '--------------------------------------------------------------------------\n\n']);

%% Tensor construction and visualization
size_tens = [10 20 30];

% Construction of a tensor with random entries
T = randn(size_tens);

% Construction of a tensor as a sum of R rank-1 terms
R = 4;
U = cpd_rnd(size_tens,R);
T = cpdgen(U);

% Construction of a tensor as a sum of R complex rank-1 terms
options = struct;
options.Real = @randn;
options.Imag = @randn;
U = cpd_rnd(size_tens,R,options);
T = cpdgen(U);

% Construction of a tensor as a multilinear transformation of a core tensor
size_core = [4 5 6];
[F,S] = lmlra_rnd(size_tens,size_core);  % F is a cell collecting the factor matrices U, V and W
Tlmlra = lmlragen(F,S);

% Perturbing a tensor
SNR = 20;
Tnoisy = noisy(T,SNR);

% Visualization
t = linspace(0,1,60).';
T = cpdgen({sin(2*pi*t+pi/4),sin(4*pi*t+pi/4),sin(6*pi*t+pi/4)});
figure(); slice3(T);
figure(); surf3(T);
figure(); voxel3(T);

U = {sin(2*pi*t+pi/4),sin(4*pi*t+pi/4),sin(6*pi*t+pi/4)};
T = noisy(cpdgen(U), 30);
visualize(U, 'original', T);

%% CPD for beginners

% Rank estimation
figure();
U = cpd_rnd([10 20 30],4);
T = cpdgen(U);
rankest(T);

% Basic CPD
R = 4;
Uest = cpd(T,R);

% Error calculation
[relerrU,P,D] = cpderr(U,Uest);
fprintf('Relative errors on the factor matrices = %e, %e and %e.\n',relerrU)
% D is a cell collecting the different scaling matrices
relerrT = frob(cpdres(T,Uest))/frob(T);
fprintf('Relative error on the reconstructed tensor = %e\n',relerrT)

%% CPD for pros

% Compression
options = struct;
options.Compression = true;
[Uest,output] = cpd(T,R,options);
fprintf('\nOutput information with default compression:')
output.Compression

options = struct;
options.Compression = false;
[Uest,output] = cpd(T,R,options);
fprintf('\nOutput information without compression:')
output.Compression

% Initialization
options = struct;
options.Initialization = @cpd_rnd;
Uest = cpd(T,R,options);

Uinit = cpd_rnd(size_tens,R);
Uest = cpd(T,Uinit);

% Core algorithms
options = struct;
options.Algorithm = @cpd_als;
Uest = cpd(T,R,options);

% Core algorithm parameters
R = 4;
T = cpdgen(cpd_rnd([10 20 30],R));
Uinit = cpd_rnd([10 20 30],R);

options = struct;
options.Compression = false;
options.Algorithm = @cpd_nls;
options.AlgorithmOptions.Display = 1;
options.AlgorithmOptions.MaxIter = 100;      % Default 200
options.AlgorithmOptions.TolFun = eps^2;     % Default 1e-12
options.AlgorithmOptions.TolX = eps;         % Default 1e-6
options.AlgorithmOptions.CGMaxIter = 20;     % Default 15
fprintf('\nExecuting CPD using non-linear least squares:\n')
[Uest_nls,output_nls] = cpd(T,Uinit,options);

options = struct;
options.Compression = false;
options.Algorithm = @cpd_als;
options.AlgorithmOptions.Display = 1;
options.AlgorithmOptions.MaxIter = 100;      % Default 200
options.AlgorithmOptions.TolFun = eps^2;     % Default 1e-12
options.AlgorithmOptions.TolX = eps;         % Default 1e-6
fprintf('\nExecuting CPD using alternating least squares:\n')
[Uest_als,output_als] = cpd(T,Uinit,options);

% Convergence plot
figure();
semilogy(output_nls.Algorithm.fval); hold all;
semilogy(output_als.Algorithm.fval);    
ylabel('Objective function'); xlabel('Iteration');
title('Convergence plot'); legend('cpd\_nls','cpd\_als')
drawnow;

%% Low multilinear rank approximation (LMLRA) and multilinear singular value decomposition (MLSVD)
size_tens = [10 20 30];
T = randn(size_tens);
[U,S,sv] = mlsvd(T);

size_core = [6 6 6];
[U,S,sv] = mlsvd(T,size_core);

[U,S] = lmlra(T,size_core);