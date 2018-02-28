require 'image'
require 'nn'

--TODO check for arguments 
img = image.load(arg[1])
net = torch.load(arg[2])

time = sys.clock() 

output = net:forward(img)
time = sys.clock - time

confidence, prediction = output:float():sort()
confidence = confidence[confidence:size(1)]
prediction = prediction[prediction:size(1)]

<<<<<<< HEAD
print('tempo di elaborazione '..time*1000..' ms')
print ('predizione = '..prediction .. ' con confidenza '..confidence)

=======

print('tempo di elaborazione '..time*1000..' ms')
print ('predizione = '..prediction .. ' con confidenza '..confidence)



>>>>>>> ece01a6c88b2ea9153728cdbf63dcf2c83a18f6a
--[[
-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
<<<<<<< HEAD
=======
    

>>>>>>> ece01a6c88b2ea9153728cdbf63dcf2c83a18f6a

    
    --here we visualize all the inputs tested and we label them with the correspective predictions 
     if opt.visualize == true then 
     for j=1,(k-1) do 
       local  img = inputs[j]
       local  predicted = model:forward(img)
       local  conf , ind = predicted:float():sort()
       print('conf -----> '..conf[conf:size(1)])
       local catgry = (ind[ind:size(1)] - 1)   -- catgry has the value of the predicted class. 
       gfx.image(img,{legend=catgry,zoom=2})
                        --image.display{image=img,legend=val,zoom=2}
       io.write("continue ([y]/n)? ")
       io.flush()
       answer = io.read() -- fittizio
       if answer =='n' then 
           break 
         end
        end 
      end


      -- confusion:
      -- test samples
      local preds = model:forward(inputs)

      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   ret = confusion.totalValid
   confusion:zero()

   return ret
end ]]
