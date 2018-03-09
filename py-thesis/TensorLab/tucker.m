size_core = [5 6 7];
[U, S] = lmlra_rnd([30,40,50], size_core);
T = lmlragen(U,S);

tic, [U1, S1] = mlsvd(T, size_core); toc
tic, [U2, S2] = mlsvd_rsi(T, size_core); toc