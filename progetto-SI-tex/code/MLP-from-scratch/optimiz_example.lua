require 'torch'
require 'optim'

torch.manualSeed(1234)
-- choose a dimension
N = 10

-- create a random NxN matrix
A = torch.rand(N, N)

-- make it symmetric positive
A = A*A:t()

-- make it definite
A:add(0.001, torch.eye(N))

-- add a linear term
b = torch.rand(N)

-- create the quadratic form
function J(x)
   return 0.5*x:dot(A*x)-b:dot(x)
end

print('Value of the function at a random point, initial state:')
x0 = torch.rand(N)
print(x0)
print('Val: ')
print(J(x0))

--xs = torch.inverse(A)*b
--print(string.format('J(x^*) = %g', J(xs)))

--define derivative
function dJ_dx(x)
   return A*x-b
end

local neval = 0
logger = optim.Logger('dummy_accuracy.log')

local feval = function(x)
   local f = J(x)
   local df_dx = dJ_dx(x)
   neval = neval + 1
   print(string.format('after %d evaluations J(x) = %f', neval, f))
   logger:add{neval, f} --,timer:time().real}

   return f,df_dx
end

config = {
   learningRate = 1e-2,
   momentum = 1e-3
   --weightDecay = 0.1
}

optim.cg(feval, x0, {maxIter = 100})
--[[
for i=1,100 do
   optim.sgd(feval, x0, config)
end
--]]
print('Optimization done.')
