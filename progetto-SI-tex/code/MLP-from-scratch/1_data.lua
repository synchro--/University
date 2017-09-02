--[[
Build & Train MLP from scratch in Torch. 
Alessio Salman    

1. Dataset
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
