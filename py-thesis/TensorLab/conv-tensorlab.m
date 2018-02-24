clc 

% mock up script to start messing around with TensorLab 

% mimic the first convolution layer 
A = randn(32,32,3); 
K = randn(3,3,3); % kernel 

F1 = convn(A, K); % N-D convolution 

%rank = rankest(A);% find the rank of the tensor wt hankelization 
disp(rank)

% config CPD options 
options = struct;
options.Algorithm = @cpd_minf;
options.Initialization = @cpd_rnd;
options.Complex = false; 

% compute the CPD 
[Uhat, out] = cpd(A, 80, options); 

% Uhat contains ... 
% out contains ... 

relerr1 = frob(cpdres(A,Uhat))/frob(A); % compute error
disp(relerr1); 

% cpdgen Generate full tensor given a polyadic decomposition.
% T = cpdgen(U) computes the tensor T as the sum of R rank-one tensors
% defined by the columns of the factor matrices Uhat{n}.
    
T1 = cpdgen(Uhat); 
A-T1 % should now be a difference in the order of 1e-15

% compute the convolution with the generated tensor 
% to see it's similar up to 1e-05 to the result F 

F2 = convn(T1, K); 
disp(frob(F1) - frob(F2)); 

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