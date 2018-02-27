function cpd_out = compute_cpd(weights_file, rank, out_file)
    clc
    disp(out_file)
    % retrieve weights 
    load(weights_file);
    W = weights{1}; 
    B = weights{2}; 
    
    % estimate best number of terms to approximate 
    % the tensor, i.e. best rank approximation 
    if(rank == 0)
        rank = rankest(W);
    end
    
    % configure options to compute CPD 
    options = struct;
    options.Compression = 'auto'
    options.Initialization = 'auto'
    options.Algorithm = @cpd_nls; % select algorithm! 
    options.AlgorithmOptions.LineSearch = @cpd_els; % Add exact line search.
    options.AlgorithmOptions.Display = 1;
    options.AlgorithmOptions.MaxIter = 200;      % Default 200
    options.AlgorithmOptions.TolFun = eps^2;     % Default 1e-12
    options.AlgorithmOptions.TolX = eps;         % Default 1e-6
    options.AlgorithmOptions.CGMaxIter = 20;     % Default 15
    
    % compute CPD 
    [cpd_out, out_nls] = cpd(W, rank, options);

    % computing the error of the reconstruction 
    res = cpdres(W, cpd_out);
    relerr = frob(cpdres(W, cpd_out))/frob(W); 

    % print error 
    fprintf('Difference in reconstruction: %e\n', res);
    fprintf('Approx err with frobenius norm: %e\n', relerr);

    cpd_s = struct 
    cpd_s.weights = cpd_out 
    cpd_s.bias = B
    save out_file cpd_s
end




%% 

%{
rank = rankest(w);

options = struct;
options.Compression = 'auto'
options.Initialization = 'auto'
options.Algorithm = @cpd_nls;
options.AlgorithmOptions.Display = 1;
options.AlgorithmOptions.MaxIter = 100;      % Default 200
options.AlgorithmOptions.TolFun = eps^2;     % Default 1e-12
options.AlgorithmOptions.TolX = eps;         % Default 1e-6
options.AlgorithmOptions.CGMaxIter = 20;     % Default 15
[Uest_nls, output_nls] = cpd(K,rank,options);

% A convergence plot can be obtained from the link in the command terminal
% output, or by plotting the objective function values
% from output.Algorithm (and output.Refinement):
figure();
grid on;
h1 = semilogy(output_nls.Algorithm.fval); hold all;
set(h1, 'LineWidth', 2)
ylabel('Objective function'); xlabel('Iteration');
legend('cpd\_nls','cpd\_nls\_rnd')
title({ "Convergence plot - init: auto") %, strcat("input=", str1, "  - Fmaps=", str2)})
%}
