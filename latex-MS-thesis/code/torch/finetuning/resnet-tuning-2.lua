require 'nn'
require 'optim'

--Fine tuning Facebook implementation of the Microsoft ResNet architecture

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)
--Load features for train and validation
t1 = torch.load('train_ants.t7').features:float()
t2 = torch.load('train_bees.t7').features:float()
Z_tr = torch.cat(t1, t2, 1) --CNN Codes
dataDim = Z_tr:size(2)

t1 = torch.load('val_ants.t7').features
t2 = torch.load('val_bees.t7').features
Z_val = torch.cat(t1, t2, 1) --CNN Codes

--1: ants
--2: bees
classes = {1,2}
lab1 = torch.Tensor(t1:size(1)):fill(1)
lab2 = torch.Tensor(t2:size(1)):fill(2)
labels_tr = torch.cat(lab1, lab2)


--1: ants
--2: bees
classes = {1,2}
lab1 = torch.Tensor(t1:size(1)):fill(1)
lab2 = torch.Tensor(t2:size(1)):fill(2)
labels_val = torch.cat(lab1, lab2)

--Shuffling the whole given dataset before dividing it in training and val sets
torch.manualSeed(42)
shuffle_tr= torch.randperm(Z_tr:size(1))
shuffle_val= torch.randperm(Z_val:size(1))

--Creating the datasets objects for the validation and training sets
validationset={}
-- we are going to use 30 % of the whole dataset as the validation set
function validationset:size()  return Z_val:size(1) end
for i=1, validationset:size() do
    validationset[i]={ Z_val[shuffle_val[i]], labels_val[shuffle_val[i]] }
end

trainingset={}
function trainingset:size()  return Z_tr:size(1) end
for i=1, trainingset:size() do
    --let's take all the images above the validationset size
    trainingset[i]={Z_tr[shuffle_tr[i]], labels_tr[shuffle_tr[i]]}
end

--- Defining the  model
model = nn.Sequential()
model:add(nn.Linear(dataDim, #classes))
model:add(nn.LogSoftMax())

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

trainer = nn.StochasticGradient(model, criterion)
trainer.hookIteration=evaluation_callback
trainer.stats={tr={},val={}} --we will use this table to save the stats
trainer.learningRate = 1e-5
trainer.maxIteration = 150 -- 150 epochs of training.
trainer.verbose=false -- we will print the stats in the callback
trainer:train(trainingset)
--save model
torch.save('tuned.t7', model)
