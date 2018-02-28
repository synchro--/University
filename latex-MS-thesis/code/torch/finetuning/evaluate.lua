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
