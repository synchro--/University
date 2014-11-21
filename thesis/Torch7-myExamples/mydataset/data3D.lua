-- data for our 3d pursuit of happiness 

----------------------------------------------------------------------
-- This script load the apple dataset used for supervised learning
-- 
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Ali Alessio Salman
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides a normalization operator


--[[ function getFiles(dir)
   local folders = {}
   local tmpfile = '/tmp/stmp.txt'
   os.execute('ls -l'..dir..' > '..tmpfile)
   local f = io.open(tmpfile)
   if not f then return files end  
   local k = 1
   for line in f:lines() do
      folders[k] = line
      k = k + 1
   end
   f:close()
   return folders
 end]] -- 

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Apple Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------

print(sys.COLORS.red ..  '==> searching datasets...')

--check that the number of folders correspond to the number of classes: 
function getNumber()
   os.execute('ls -l |grep -v "total" | wc -l > dirCount')
   local f = io.open('dirCount')
   count = f:line()
   f:close()
   return count
end 

  function getNames(dir)
   local folders = {}
   local tmpfile = 'list.txt'
   os.execute('ls -l | grep -v total'..dir..' > '..tmpfile)
   local f = io.open(tmpfile)
   if not f then return files end  
   local k = 1
   for line in f:lines() do
      folders[k] = line
      k = k + 1
   end
   f:close()
   return folders
 end
 
 function getNumber()
   local count=0
   os.execute('
 
 --classes GLOBAL VAR
classes = {'person','tree','stairs','ecc'}
local count = getNumber('.') -- recupero il numero delle subdirectories quindi dei dataset 

if count != #classes
   print ('missing some data... exit')
   os.exit(-1)
end 

folders = getNames('.')

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> loading dataset')

torch.setdefaulttensortype('torch.FloatTensor') -- preprocessing requires a floating point representation 

--we load data from disk 
local total = 5000
local imagesAll = torch.Tensor(total,3,32,32)
local labelsAll = torch.Tensor(total)

os.execute('cd' ..folders[i])
   
--function ok

for i=1,#classes do 
   --load images
   local count = getNumber('.') -- retrieving total number of photos 
   for j=1,count do 
      imagesAll[{  {(i*count), (i*count+j)}  }] = image.load(folders[j]..j..'.png') --person.1.png, person.2.png
      labelsAll[{  {(i*count), (i*count+j)}  }] = classes[i] --lega l'indice alla label che indica la classe e ogni indice corrisponde alla particolare immagine 
end 
  os.execute('cd ../'..folders[i+1]) -- next directory 
end  

-- shuffle dataset: get shuffled indices in this variable:
local labelsShuffle = torch.randperm((#labelsAll)[1]) -- mescola gli indici ma mantiene le corrispondenze: 
--[[ anzichÃ¨ avere una corrispondenza ordinata, la si mescola : 
     1-apple               23-apple          
     2-apple               17599-bg
     .          ==>        .
     .                     .
     1200 - bg             900-apple
     1201 - bg             7-apple
     1202 - bg             1665-bg 
]]-- 


local portionTrain = 0.8 -- 80% is train data, rest is test data
local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = labelsShuffle:size(1) - trsize

--create train set 
trainData {
    data = torch.Tensor(trsize,1,32,32) -- yuv o solo y 
    labels  = torch.Tensor(trsize)
    size = function() size return trsize end   
} 

--create test set
testData = {
      data = torch.Tensor(tesize, 1, 32, 32),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }


-- now we store data in the correspondent Tensors
for i=1,trsize do
   trainData.data[i] = imagesAll[labelsShuffle[i]][1]:clone() -- shuffled training imgs
   trainData.labels[i] = labelsAll[labelsShuffle[i]]  -- shuffled lables for training 
end
for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]][1]:clone() -- same as above 
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end

---------------------------------------------------------------------------------------
-- PREPROCESSING DATA -- 
print(sys.COLORS.red ..  '==> preprocessing data')

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

local channels = {'y','u','v'}

--convert all to yuv

for i=1,trsize do 
  trainData.data[i] = image.rgb2yuv(trainData.data[i])
  end 

for i=1,tesize do 
  testData.data[i] = image.rgb2yuv(testData.data[i])
  end 

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.

print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do 
 mean = trainData[{ {},i,{},{} }]:mean()
 std  = trainData[{ {},i,{},{} }]:std() 
 trainData[{ {},i,{},{} }] = trainData[{{},i,{},{}]:add(-mean[i]) -- syntax matlab-like: [{ {1dim} {2dim} {3dim} {4dim} }]
 trainData[{ {},i,{},{} }] = trainData[{{},i,{},{}]:div(std[i])
end 

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

--define Normalization neighborhood 
neighborhood = image.gaussian1D(5) -- vedere come decidere questo parametro 
norm = nn.SpatialContrastiveNormalization(1,neighborhood,1):float() --1 input (canale y) 

--normalize each channel locally 
for channel in ipairs(channels)
for i=1,trsize do 
  trainData[{ i,{channel},{},{} }] = norm:forward(trainData[{ i,{channel},{},{} }])
  end 

for i=1,tesize do 
    tesize[{ i,{c},{},{} }] = norm:forward(tesize[{ i,{c},{},{} }])
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
  
print(sys.COLORS.red ..  '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local first256Samples_y = trainData.data[{ {1,256},1 }]
   gfx.image(first256Samples_y, {nrow=16, legend='Some training examples: Y channel'})
   local first256Samples_y = testData.data[{ {1,256},1 }]
   gfx.image(first256Samples_y, {nrow=16, legend='Some testing examples: Y channel'})
   --image.display{image=first256Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end


-- return values! 
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}