----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem.
--
-- Illustra diversi punti:
-- 1. descrizione del modello
-- 2. scelta della loss function da minimizzare
-- 3. creazione del dataset come semplice Lua table
-- 4. definizione di procedure di training e testing
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'gfx.js'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'logs', 'subdirectory to save/log experiments in')
cmd:option('-network', 'logs/cifar.net', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-visualize',false,'plotting progress on training and testing')
cmd:option('-browser',false,'visualize testing in the browser')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
---
classes = {'airplane', 'car', 'bird', 'cat',
   'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

normkernel = image.gaussian1D(13)

if opt.network == '' then
   --first, we get a sequential model onto which we'll add every layer
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network
      ------------------------------------------------------------
      -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
      model:add(nn.ReLU())
      model:add(nn.SpatialLPPooling(16,2,2, 2, 2, 2))
      model:add(nn.SpatialSubtractiveNormalization(16, normkernel))

      -- stage 2 : filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 256, 4), 5, 5))
      model:add(nn.ReLU())
      model:add(nn.SpatialLPPooling(256,2,2, 2, 2, 2))
      model:add(nn.SpatialSubtractiveNormalization(256, normkernel))

      -- stage 3 : standard 2-layer neural network
      model:add(nn.Reshape(256*5*5))
      model:add(nn.Linear(256*5*5, 128))
      model:add(nn.ReLU())
      model:add(nn.Linear(128,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32, 1*32*32))
      model:add(nn.Tanh())
      model:add(nn.Linear(1*32*32, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(3*32*32))
      model:add(nn.Linear(3*32*32,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<cifar> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   trsize = 50000
   tesize = 10000
else
   trsize = 2000
   tesize = 1000
end

-- download dataset
if not paths.dirp('cifar-10-batches-t7') then
   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
   local tar = paths.basename(www)
   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
end

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
--labels must go from 1 to 10
trainData.labels = trainData.labels + 1

subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<trainer> preprocessing data (color space + normalization)'

-- preprocess trainSet
normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
   --rgb -> yuv
   local rgb = trainData.data[i]
   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[1] = normalization(yuv[{{1}}])
   trainData.data[i] = yuv
end
-- normalize u globally:
mean_u = trainData.data[{ {},2,{},{} }]:mean()
std_u = trainData.data[{ {},2,{},{} }]:std()
trainData.data[{ {},2,{},{} }]:add(-mean_u)
trainData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
mean_v = trainData.data[{ {},3,{},{} }]:mean()
std_v = trainData.data[{ {},3,{},{} }]:std()
trainData.data[{ {},3,{},{} }]:add(-mean_v)
trainData.data[{ {},3,{},{} }]:div(-std_v)

-- preprocess testSet
for i = 1,testData:size() do
   -- rgb -> yuv
   local rgb = testData.data[i]
   --local yuv = trainData.data[i]

   local yuv = image.rgb2yuv(rgb)
   -- normalize y locally:
   yuv[{1}] = normalization(yuv[{{1}}])
   testData.data[i] = yuv
end
-- normalize u globally:
testData.data[{ {},2,{},{} }]:add(-mean_u)
testData.data[{ {},2,{},{} }]:div(-std_u)
-- normalize v globally:
testData.data[{ {},3,{},{} }]:add(-mean_v)
testData.data[{ {},3,{},{} }]:div(-std_v)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- function to visualize the Network weights (few of them) during training
function display(input)
   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter%10 == 0 then
      if opt.model == 'convnet' then
         win_w1 = image.display{image=model:get(2).weight, zoom=4, nrow=10,
            min=-1, max=1,
            win=win_w1, legend='stage 1: weights', padding=1}
         win_w2 = image.display{image=model:get(6).weight, zoom=4, nrow=30,
            min=-1, max=1,
            win=win_w2, legend='stage 2: weights', padding=1}
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{image=W1, zoom=0.5,
            min=-1, max=1,
            win=win_w1, legend='W1 weights'}
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{image=W2, zoom=0.5,
            min=-1, max=1,
            win=win_w2, legend='W2 weights'}
      end
   end
   iter = iter + 1
end

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local input = dataset.data[i]
         local target = dataset.labels[i]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      -- i.e. forward and backward pass
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f = 0

         -- evaluate function for complete mini batch
         for i = 1,#inputs do
            -- estimate f
            local output = model:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            --backpropagation
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])

            -- visualize weights
            if opt.visualize then
             display(inputs[i])
            end
         end

         -- normalize gradients and f(X)
         gradParameters:div(#inputs)
         f = f/#inputs

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
            maxIter = opt.maxIter,
            nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
            weightDecay = opt.weightDecay,
            momentum = opt.momentum,
            learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
            t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'cifar.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local input = dataset.data[t]
      local target = dataset.labels[t]

      --code to visualize examples in the browser using gfx.js
      if(opt.browser) then
         -- test sample
         myTime = sys.clock()

         local pred = model:forward(input) -- predicted
         local conf , index = pred:float():sort() -- sort restituisce i risultati ordinati in ordine decrescente
         --taking time
         myTime = sys.clock() - myTime

         local catgry = (index[index:size(1)]) -- prendo il valore all'ultimo indice di index
         catgry = classes[catgry] -- catgry has the value of the predicted class.
         label = catgry..'. Test time: '..math.floor((myTime*1000))..'ms'

         --visualizziamo nel browser
         gfx.image(input,{legend=label,zoom=5})

         io.write("continue ([y]/n)? ")
         io.flush()
         answer = io.read() -- se n ==> break

         if(answer == 'n') then
            break
         end

      else
         local pred = model:forward(input) -- predicted
         confusion:add(pred, target)

      end --if/else

   end --for

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
end -- function

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   if not opt.browser then
      complete = train(trainData)
      test(testData)

      -- plot errors
      if opt.plot then
         trainLogger:style{['% mean class accuracy (train set)'] = '-'}
         testLogger:style{['% mean class accuracy (test set)'] = '-'}
         trainLogger:plot()
         testLogger:plot()
      end

      if complete == 0.8 then
         break
      end

   else -- se vogliamo solo visualizzare le immagini testate, saltiamo il training
      test(testData)

   end

end

print ('training completed, with: '..(complete*100)..'% accuracy')
