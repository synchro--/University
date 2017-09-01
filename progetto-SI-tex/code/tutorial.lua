--hyper-parameters
nfeats = 3               --3D input volume
nstates = {16, 256, 128} --output at each level
filtsize = 5             --filter size or kernel
poolsize = 2

------------------------------------------------------------
-- convolutional network
------------------------------------------------------------
-- stage 1 :  filter bank -> squashing -> max pooling -> mean+std normalization
--            3 -> 16
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> max pooling -> mean+std normalization
--           16 -> 256
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))
model:add(nn.SpatialSubtractiveNormalization(256, normkernel))

-- stage 3 : standard 2-layer neural network
--           256 -> 128 -> 10 -> classification
model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nstates[3],#classes))
