require 'nn'
require 'csvigo' ---luarocks install csvigo
require 'torch'
require 'evaluate'
local opts = require 'opts'
opt = opts.parse(arg)

--load model and save prediction on unseen test data
model = torch.load(opt.save..'tuned.t7')
t1 = torch.load(opt.data..'val_ants.t7').features
t2 = torch.load(opt.data..'val_bees.t7').features
Z = torch.cat(t1,t2,1)

labels = torch.Tensor(t1:size(1)):fill(1)
lab2 = torch.Tensor(t2:size(1)):fill(2)
labels = torch.cat(labels, lab2)

dataset={}
function dataset:size() return Z:size(1)  end

for i=1, dataset:size() do
    dataset[i]={Z[i]:float()}
end

pred,_ = eval(model, dataset, false)

output={}
correct = 0
for i=1,#pred do
  if labels[i] == pred[i] then
    correct = correct + 1
  end
  output[i]={id=i,Label=pred[i]}
end
accuracy = correct / labels:size(1)
print('Testing on '..labels:size(1)..' unseen examples...')
print(string.format('#Test Accuracy: %.3f',accuracy))

csvigo.save('test.csv', output, true)
