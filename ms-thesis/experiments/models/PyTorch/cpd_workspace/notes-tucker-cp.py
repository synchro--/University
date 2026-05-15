# Così facciamo il procedimento del paper di Astari li come cazzo ti chiami 

In [151]: weights = np.abs(np.random.randn(64,32,3,3)) 

In [152]: w_cp = weights.reshape(64, 32, 9) # not necessary for VBMF but for CP yes! 

In [316]: first, second, third = parafac(w_cp, rank=10, init='svd') 

In [317]: first.shape
Out[317]: (64, 10)

In [318]: second.shape
Out[318]: (64, 10)

In [319]: third.shape
Out[319]: (9, 10)

In [153]: u0 = tl.base.unfold(w_cp, 0) 

In [154]: u1 = tl.base.unfold(w_cp, 1) 

In [155]: u2 = tl.base.unfold(w_cp, 2) 

In [156]: u2.shape
Out[156]: (9, 2048)

In [157]: u1.shape
Out[157]: (32, 576)

In [158]: u0.shape
Out[158]: (64, 288)

In [159]: _, diag_0, _, _ = VBMF.EVBMF(u0)

In [160]: _, diag_2, _, _ = VBMF.EVBMF(u1)

In [161]: _, diag_1, _, _ = VBMF.EVBMF(u1)

In [162]: _, diag_2, _, _ = VBMF.EVBMF(u2)

In [163]: diag_0.shape
Out[163]: (1, 1)

In [164]: diag_1.shape
Out[164]: (1, 1)

In [165]: diag_2.shape
Out[165]: (1, 1)





############# SVD Faster-RCNN ##################
def compress_weights(W, l):
    """Compress the weight matrix W of an inner product (fully connected) layer
    using truncated SVD.
    Parameters:
    W: N x M weights matrix
    l: number of singular values to retain
    Returns:
    Ul, L: matrices such that W \approx Ul*L
    """

    # numpy doesn't seem to have a fast truncated SVD algorithm...
    # this could be faster
    U, s, V = np.linalg.svd(W, full_matrices=False)

    Ul = U[:, :l]
    sl = s[:l]
    Vl = V[:l, :]

    L = np.dot(np.diag(sl), Vl)
    return Ul, L

# qui ritorna un 2 array approssimati, facciamo a K singular values allora: 

fc1 = (Ul.shape[1], Ul.shape[0]) ==> fc1.weight.data = Ul.T 
fc2 = (L.shape[1], L.shape[0]) ==> same thing