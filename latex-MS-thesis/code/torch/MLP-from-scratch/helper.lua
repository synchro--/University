--Torch Helper functions

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
