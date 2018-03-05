%% Rankest performance evaluation 
%{
Running rankest(T) on a dense, sparse or incomplete tensor T plots an L-curve which represents the balance
between the relative error of the CPD and the number of rank-one terms R. 

The lower bound is based on the truncation error of the tensors multilinear singular values 
For incomplete and sparse tensors, this lower bound is not available and the first value to be tried for R is 1. 
The number of rank-one terms is increased until the relative error of the approximation is less than options.MinRelErr. 
In a sense, the corner of the resulting L-curve makes an optimal trade-off between accuracy and
compression. The rankest tool computes the number of rank-one terms R corresponding to
the L-curve corner and marks it on the plot with a square. This optimal number of rank-one
terms is also rankests first output.
%}

A = {}
times = []
ranks = []
n_inputs = [3, 9, 12, 32, 64]
n_outputs = [12, 32, 64, 128, 256]
dim = 3; 

for i=1:5
    A{i} = abs(randn(n_inputs(i), dim, dim, n_outputs(i)))
    tic
    ranks(i) = rankest(A{i})
    times(i) = toc
end 

disp('estimated rankest times:') 
disp(times)
 
%mul_dims = n_inputs .* n_outputs;
mul_dims = n_inputs + n_outputs;  

h3 = figure(3);
h3 = plot(mul_dims(1:end-1), times);
set(h1, 'LineWidth',3);
grid on

title('Rankest computation time') 
xlabel('(N. of Inputs) x (N. of Outputs)') 
ylabel('Rank estimation in SECONDS') 


%% plots 
h1 = figure(1); 
h1 = plot(n_inputs, times);
set(h1, 'LineWidth',3);
grid on

title('Rankest computation time') 
xlabel('Number of INPUTS') 
ylabel('Rank estimation in SECONDS') 

h2 = figure(2); 
h2 = plot(n_outputs, times); 
set(h2, 'LineWidth',3);
grid on

title('Rankest computation time') 
xlabel('Number of OUTPUTS') 
ylabel('Rank estimation in SECONDS') 

h3 = figure(3);
h3 = plot(mul_dims, times);
set(h1, 'LineWidth',3);
grid on

title('Rankest computation time') 
xlabel('(N. of Inputs) x (N. of Outputs)') 
ylabel('Rank estimation in SECONDS') 



