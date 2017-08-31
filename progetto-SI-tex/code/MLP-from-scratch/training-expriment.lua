--[[
This file build and trains the network using different optimization techniques.
The techniques are hardcoded in the "techniques" list.
For each technique the training is performed from scratch and iterated a maximum number of times.
Results of each training are logged in the respective log file (i.e. SGD.log, ADAM.log, ...)
--]]

require 'partSix'

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Optimization Method')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-optimization', 'SGD', 'type of optimization method to use to minimize the cost function (CG, SGD, LBFGS, NAG...)')
   cmd:option('-precision', 'torch.FloatTensor', 'Tensor precision [FloatTensor]')
   cmd:text()
   opt = cmd:parse(arg or {})
end

--optimization configs
techniques = {'SGD', 'NAG', 'ADAM', 'LBFGS'}
opt.learningRate = 1e-1
opt.maxIter = 200
opt.weightDecay = 0
opt.momentum = 0.0

--take an opt table and configure parameters depending on the
--optimization method in use. Returns the opt method chosen
function configOpt(opt)
   if opt.optimization == 'CG' then
      optimState = {
         maxeval = opt.maxIter,
         maxIter = opt.maxIter,
         verbose = true
      }
      optimMethod = optim.cg

   elseif opt.optimization == 'LBFGS' then
      optimState = {
         learningRate = opt.learningRate,
         maxIter = opt.maxIter,
         nCorrection = 10
      }
      optimMethod = optim.lbfgs

   elseif opt.optimization == 'SGD' then
      optimState = {
         learningRate = opt.learningRate,
         weightDecay = opt.weightDecay,
         momentum = opt.momentum,
         learningRateDecay = 1e-7
      }
      optimMethod = optim.sgd

   elseif opt.optimization == 'ASGD' then
      optimState = {
         eta0 = opt.learningRate,
         t0 = 1
      }
      optimMethod = optim.asgd

   elseif opt.optimization == 'NAG' then
      optimState = {
         learningRate = opt.learningRate,
         weightDecay = opt.weightDecay,
         momentum = 0.3,
         learningRateDecay = 1e-7
      }

      optimMethod = optim.nag

   elseif opt.optimization == 'ADAM' then
      optimState = {
         learningRate = opt.learningRate,
         beta1= 0.9,
         beta2= 0.999,
         epsilon= 1e-08
      }

      optimMethod = optim.adam

   else print("Error: UNKNOWN optimization method")
   end

   return optimMethod
end

nn = Neural_Network(2,3,1)
nn:forward(X)
tr = Trainer(nn)
init_params = nn:getParams()

--[[
--Training and logging for only one opt method

logfile = opt.optimization..'.log'
logger = optim.Logger(logfile)
optimMethod = configOpt(opt)
tr:train(X,y)
--]]

--for each optimization techniques contained in the homonym list
--re-initialize the starting parameters
--then, train from scratch and log results in a log file

for _,name in ipairs(techniques) do
   --reset initial state for comparison with the next algorithm
   nn:setParams(init_params)

   logfile = name..'.log'
   logger = optim.Logger(logfile)
   opt.optimization = name
   --configura parameters for the specified method
   optimMethod = configOpt(opt)
   tr:train(X,y)
end
