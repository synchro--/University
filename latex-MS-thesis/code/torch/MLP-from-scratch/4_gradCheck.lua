--[[
--------------------------------------
Build & Train MLP from scratch in Torch.
Alessio Salman
---------------------------------------

4. Numerical Gradient Check
--]]

----------------------- Part 1 ----------------------------
th = require 'torch'
bestProfit = 6.0

--bad written, just a way to normalize divide every column for its max value
function normalizeTensorAlongCols(tensor)
   local cols = tensor:size()[2]
   for i=1,cols do
      tensor[{ {},i }]:div(tensor:max(1)[1][i])
   end
end

-- X = (num coperti, ore di apertura settimanali), y = profitto lordo annuo in percentuale
torch.setdefaulttensortype('torch.DoubleTensor')
X = th.Tensor({{22,42}, {25,38}, {30,40}})
y = th.Tensor({{2.8},{3.4},{4.4}})

--normalize
normalizeTensorAlongCols(X)
y = y/bestProfit

----------------------- Part 4 ----------------------------
--creating class NN in Lua, using a nice class utility
require 'class'

--init NN
Neural_Network = class(function(net, inputs, hiddens, outputs)
      net.inputLayerSize = inputs
      net.hiddenLayerSize = hiddens
      net.outputLayerSize = outputs
      net.W1 = th.randn(net.inputLayerSize, net.hiddenLayerSize)
      net.W2 = th.randn(net.hiddenLayerSize, net.outputLayerSize)
   end)

--define a forward method
function Neural_Network:forward(X)
   --Propagate inputs though network
   self.z2 = th.mm(X, self.W1)
   self.a2 = th.sigmoid(self.z2)
   self.z3 = th.mm(self.a2, self.W2)
   yHat = th.sigmoid(self.z3)
   return yHat
end

function Neural_Network:d_Sigmoid(z)
   --Gradient of sigmoid
   return th.exp(-z):cdiv( (th.pow( (1+th.exp(-z)),2) ) )
end

function Neural_Network:costFunction(X, y)
   --Compute the cost for given X,y, use weights already stored in class
   self.yHat = self:forward(X)
   --NB torch.sum() isn't equivalent to python sum() built-in method
   --However, for 2D arrays whose one dimension is 1, it won't make any difference
   J = 0.5 * th.sum(th.pow((y-yHat),2))
   return J
end

function Neural_Network:d_CostFunction(X, y)
   --Compute derivative wrt to W and W2 for a given X and y
   self.yHat = self:forward(X)
   delta3 = th.cmul(-(y-self.yHat), self:d_Sigmoid(self.z3))
   dJdW2 = th.mm(self.a2:t(), delta3)

   delta2 = th.mm(delta3, self.W2:t()):cmul(self:d_Sigmoid(self.z2))
   dJdW1 = th.mm(X:t(), delta2)

   return dJdW1, dJdW2
end

--Helper Functions for interacting with other classes:
function Neural_Network:getParams()
   --Get W1 and W2 unrolled into a vector
   params = th.cat((self.W1:view(self.W1:nElement())), (self.W2:view(self.W2:nElement())))
   return params
end

function Neural_Network:setParams(params)
   --Set W1 and W2 using single paramater vector.
   W1_start = 1 --index starts at 1 in Lua
   W1_end = self.hiddenLayerSize * self.inputLayerSize
   self.W1 = th.reshape(params[{ {W1_start, W1_end} }], self.inputLayerSize, self.hiddenLayerSize)

   W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
   self.W2 = th.reshape(params[{ {W1_end+1, W2_end} }], self.hiddenLayerSize, self.outputLayerSize)
end

function Neural_Network:computeGradients(X, y)
   --returns the weights gradients as a one-rolled vector
   dJdW1, dJdW2 = self:d_CostFunction(X, y)
   return th.cat((dJdW1:view(dJdW1:nElement())), (dJdW2:view(dJdW2:nElement())))
end

function computeNumericalGradient(NN, X, y)
   paramsInitial = NN:getParams()
   numgrad = th.zeros(paramsInitial:size())
   perturb = th.zeros(paramsInitial:size())
   e = 1e-4

   for p=1,paramsInitial:nElement() do
      --Set perturbation vector
      perturb[p] = e
      NN:setParams(paramsInitial + perturb)
      loss2 = NN:costFunction(X, y)

      NN:setParams(paramsInitial - perturb)
      loss1 = NN:costFunction(X, y)

      --Compute Numerical Gradient
      numgrad[p] = (loss2 - loss1) / (2*e)

      --Return the value we changed to zero:
      perturb[p] = 0
   end

   --Return Params to original value:
   NN:setParams(paramsInitial)
   return numgrad
end

--test if we actually make the calculations correctly
NN = Neural_Network(2,3,1)

print('Gradient checking...')
numgrad = computeNumericalGradient(NN, X, y)
grad = NN:computeGradients(X, y)
--[[
In order to make an accurate comparison of the 2 vectors
we can calculate the difference as the ratio of:
numerator  --> the norm of the difference
denumerator--> the norm of the sum
Should be in the order of 10^-8 or less
--]]
diff = th.norm(grad-numgrad)/th.norm(grad+numgrad)
print(string.format('The difference is %e',diff))
