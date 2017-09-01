require 'nn'
require 'optim'
local opts = require 'opts'
--retrieving command-line options
local opt = opts.parse(arg)

--Fine tuning Facebook implementation of the Microsoft ResNet architecture
print(opt.tensorType)
torch.setdefaulttensortype(opt.tensorType)
torch.setnumthreads(opt.nThreads)
--Load features for training
t1 = torch.load(opt.data..'train_ants.t7').features:float()
t2 = torch.load(opt.data..'train_bees.t7').features:float()
Z = torch.cat(t1, t2, 1) --CNN Codes
dataDim = Z:size(2)

--1: ants
--2: bees
classes = {1,2}
lab1 = torch.Tensor(t1:size(1)):fill(1)
lab2 = torch.Tensor(t2:size(1)):fill(2)
labels = torch.cat(lab1, lab2)

--Shuffling the whole given dataset before dividing it in training and val sets
torch.manualSeed(opt.manualSeed)
shuffle= torch.randperm(Z:size(1))

--Creating the datasets objects for the validation and training sets

validationset={}
-- we are going to use 30 % of the whole dataset as the validation set
-- the ratio can be changed using the validationSize option
function validationset:size() return opt.validationSize * Z:size(1) end
for i=1, validationset:size() do
   validationset[i]={Z[shuffle[i]], labels[shuffle[i]]}
end

trainingset={}
function trainingset:size() return Z:size(1) - validationset:size() end
for i=1, trainingset:size() do
   trainingset[i]={Z[shuffle[ validationset:size()+ i ]], labels[shuffle[ validationset:size()+i ]]}
end

--- Defining the model
if opt.resume ~= 'none' then
   model = torch.load(opt.save..'tuned.t7')
else
   model = nn.Sequential()
   model:add(nn.Linear(dataDim, #classes))
   model:add(nn.LogSoftMax())
end

--Defining the loss function
criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(model, criterion)

--Defining the evaluation function
function eval(model, dataset, validation)
   local correct=-1
   local r={}
   for i=1, dataset:size() do
      local example=dataset[i]
      local img = example[1]
      local label = example[2]
      local prediction= model:forward(img) --this output the prob (class \| image)
      local confidences, indices = torch.sort(prediction, true) -- let's sort the prob
      r[i]=indices[1] -- Picking up the class with highest confidence
      if validation then --If this is the validation set we can estimate the accuracy
         if r[i]==label then
            correct=correct+1
         end
      end
   end
   return r, correct
end

print('Start training, using an evaluation_callback function...')
trainLogFile = 'training.log'
logger = optim.Logger(trainLogFile)

local function evaluation_callback(trainer, iteration, currentError)
   _, correct=eval(trainer.module, trainingset, true)
   training_acc= correct / trainingset:size()
   print("# test accuracy = " .. training_acc)

   _, correct=eval(trainer.module, validationset, true)
   acc = correct / validationset:size()
   print("# validation accuracy = " .. acc)

   --save values to be logged later on
   if trainer.stats then
      logger:add{iteration, training_acc, acc}
      trainer.stats.tr[iteration]=training_acc
      trainer.stats.val[iteration]=acc
   end
end

--define the trainer parameters
trainer = nn.StochasticGradient(model, criterion)
trainer.hookIteration=evaluation_callback --link callback function
trainer.stats={tr={},val={}} --we will use this table to save the stats
trainer.learningRate = opt.LR
trainer.maxIteration = opt.nEpochs -- epochs of training
trainer.verbose = false -- print stats in the callback
--let's train
trainer:train(trainingset)
--save model
torch.save(opt.save..'tuned.t7', model)
