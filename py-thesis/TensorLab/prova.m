clc
clear

% mock up script to start messing around with TensorLab

% mimic the first convolution layer
A = randn(28, 28, 128);
K = randn(3,3,192); % kernel

F1 = convn(A, K); % N-D convolution

% find the rank of the tensor (NP-Hard problem)
% trial-and-error to find best tradeoff between:
% compression vs. accuracy

rank = rankest(K);
disp(rank)

options = struct;
options.Compression = 'auto'
options.Initialization = 'auto'
options.Algorithm = @cpd_nls;
options.AlgorithmOptions.Display = 1;
options.AlgorithmOptions.MaxIter = 100;      % Default 200
options.AlgorithmOptions.TolFun = eps^2;     % Default 1e-12
options.AlgorithmOptions.TolX = eps;         % Default 1e-6
options.AlgorithmOptions.CGMaxIter = 20;     % Default 15
[Uest_nls,output_nls] = cpd(K,rank,options);

options = struct;
options.Compression = 'auto'
options.Initialization = 'auto'
options.Algorithm = @cpd_als;
options.AlgorithmOptions.Display = 1;
options.AlgorithmOptions.MaxIter = 100;      % Default 200
options.AlgorithmOptions.TolFun = eps^2;     % Default 1e-12
options.AlgorithmOptions.TolX = eps;         % Default 1e-6
[Uest_als,output_als] = cpd(K,rank,options);

% A convergence plot can be obtained from the link in the command terminal
% output, or by plotting the objective function values
% from output.Algorithm (and output.Refinement):
figure();
h1 = semilogy(output_nls.Algorithm.fval); hold all;
h2 = semilogy(output_als.Algorithm.fval);
set(h1, 'LineWidth', 2)
set(h2, 'LineWidth', 2)
ylabel('Objective function'); xlabel('Iteration');
title({'Convergence plot - init: auto', 'Input = 28x28x128 - FMaps = 3x3x192'}); legend('cpd\_nls','cpd\_als')


%{
config CPD options
options = struct;
options.Algorithm = @cpd_als;
options.Initialization = @cpd_rnd;
options.Complex = false;
%}



% compute the CPD
% [Uhat, out] = cpd(A, rank, options);

% Uhat contains ...
% out contains ...

relerr1 = frob(cpdres(K,Uest_als))/frob(K); % compute error
disp(relerr1);

relerr2 = frob(cpdres(K,Uest_nls))/frob(K); % compute error
disp(relerr2);

% cpdgen Generate full tensor given a polyadic decomposition.
% T = cpdgen(U) computes the tensor T as the sum of R rank-one tensors
% defined by the columns of the factor matrices Uhat{n}.

T1 = cpdgen(Uest_nls);
Diff = K-T1; % should now be a difference in the order of 1e-15

% compute the convolution with the generated tensor
% to see it's similar up to 1e-05 to the result F

F2 = convn(A, T1);
disp(frob(F1) - frob(F2));

%{
%% Now for the Kernel %%

rank = rankest(K);
[U, out] = cpd(K, rank);
T2 = cpdgen(U);

fprintf("Difference T-Kernel is %e\n", T2-K);
relerr2 = frob(cpdres(K, U))/frob(K);
fprintf("Approx. err with frobenius norm is %e\n", relerr2);

F2 = convn(A, T2);
fprintf("frob difference btw F1 F2 is: %e\n", frob(F1) - frob(F2));

%% Now both

F3 = convn(T1, T2);
fprintf("frob difference btw original conv F1 and Approx. conv F3 is: %e\n", frob(F1) - frob(F3));
%}