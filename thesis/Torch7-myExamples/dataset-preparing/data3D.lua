----------------------------------------------------------------------
-- This script loads a datasets composed of N classes, N = number of folders in the directory
-- It should be run in the parent directory that contains all the other folders of pictures
-- for example, folders like: */parent/person/ */parent/tree */parent/cat */parent/car ecc
-- ==> run the script in parent. 'th path/to/parent/data3D.lua

-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- A.A.Salman
----------------------------------------------------------------------

require 'torch' -- torch
require 'image' -- for color transforms
require 'gfx.js' -- to visualize the dataset
require 'nn' -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-img','png','type of image: png | jpg | others') -- th data3D.lua -img png/jpg
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------

print(sys.COLORS.red .. '> Using '..opt.img..' format')

print(sys.COLORS.red .. '==> searching datasets...')

--check number of files in a directory
function getNumber(dir)
   count={}
   tmpfile='count.txt'
   os.execute("ls -l " ..dir.." | grep -v total | wc -l > "..tmpfile)
   local f = io.open(tmpfile)
   local k = 1
   for line in f:lines() do
      count[k] = line
      k = k + 1
   end
   f:close()
   return count[1]
end

--gets the name of all the files in a directory
function getNames(dir)
   local folders = {}
   os.execute('rm list.txt')
   local tmpfile='list.txt'
   --os.execute("ls -l "..dir.." | grep -v total | awk -F' ' '{print $9'}> "..tmpfile)
   os.execute("for i in "..dir.."/*; do if test -d $i; then echo $(basename $i) >>" ..tmpfile.."; fi; done")
   local f = io.open('list.txt')
   if not f then return files end
   local k = 1
   for line in f:lines() do
      folders[k] = line
      k = k + 1
   end
   f:close()
   return folders
end

--check the 3-dimensions of the input images
function checkImg(img)
   size = img:size()
   if (size[1] ~= 3) or size[2]~=size[3] or size[2]~=32 then
      --facciamo il resizing
      newImg = image.scale(img,32,32)
      return newImg
   end

   return img
end

--qui si definisce un array con i nomi di tutte le classi del problema
--classes GLOBAL VAR
classes = {'person','tree','stairs','ecc'} -- change ecc with the name of all the classes
local count = getNumber('.') -- recupero il numero delle subdirectories quindi delle classi del dataset

folders = getNames('.') --recupero il nome di ogni directory dal quale prendere le immagini

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> loading dataset')

torch.setdefaulttensortype('torch.FloatTensor') -- preprocessing requires a floating point representation

--we load data from disk
--local total = 5000
local total = 36
local imagesAll = torch.Tensor(total,3,32,32)
local labelsAll = torch.Tensor(total)

--Esempio con immagini 'png', ma si può cambiare decidendo da riga di comando il tipo
for i=1,#classes do
   --load images
   prefix=folders[i]..'/'..folders[i] --prefix = person/person
   local count = getNumber(folders[i]) -- retrieving total number of photos
   local offset = count*(i-1) -- si parte da qui per non sovrascrivere le immagini già caricate. Al primo loop è 0
   for j=1,count do
      local img = image.load(prefix..'.'..j..'.'..opt.img) --person/person.1.png, person/person.2.png
      img = checkImg(img)
      imagesAll[offset + j] = img
      labelsAll[offset + j] = i --lega l'indice alla label che indica la classe e ogni indice corrisponde alla particolare immagine
   end
end

-- shuffle dataset: get shuffled indices in this variable:
local labelsShuffle = torch.randperm((#labelsAll)[1]) -- mescola gli indici ma mantiene le corrispondenze:

--[[ dopo si otterrà una corrispondenza ordinata ma mescolata
1-apple 23-apple
2-apple 17599-bg
. ==> .
. .
1200 - bg 900-apple
1201 - bg 7-apple
1202 - bg 1665-bg
]]--

local portionTrain = 0.8 -- 80% is train data, rest is test data
local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = labelsShuffle:size(1) - trsize
print('size: '..trsize..' '..tesize)

--create train set
trainData = {
   data = torch.Tensor(trsize,1,32,32), -- yuv o solo y
   labels = torch.Tensor(trsize),
   size = function() return trsize end
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
   trainData.labels[i] = labelsAll[labelsShuffle[i]] -- shuffled lables for training
end

for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]][1]:clone() -- same as above
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end

---------------------------------------------------------------------------------------
-- PREPROCESSING DATA --
print(sys.COLORS.red .. '==> preprocessing data')

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
-- + images are mapped into YUV space, to separate luminance information
-- from color information
-- + the luminance channel (Y) is locally normalized, using a contrastive
-- normalization operator: for each neighborhood, defined by a Gaussian
-- kernel, the mean is suppressed, and the standard deviation is normalized
-- to one.
-- + color channels are normalized globally, across the entire dataset;
-- as a result, each color component has 0-mean and 1-norm across the dataset.

--local channels = {'y','u','v'}
local channels = {'y'}

--convert all to yuv, this step can be undone if you prefer to keep it similar to our biological process
--N.B. Si può ovviamente fare solo con immagini a colori

--[[for i=1,trainData:size() do
trainData.data[i] = image.rgb2y(trainData.data[i])
end

for i=1,testData:size() do
testData.data[i] = image.rgb2yuv(testData.data[i])
end ]]

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.

print(sys.COLORS.red .. '==> preprocessing data: normalize each feature (channel) globally')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   print(i)
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }] = trainData.data[{ {},i,{},{} }]:add(-mean[i]) -- syntax matlab-like: [{ {1dim} {2dim} {3dim} {4dim} }]
   trainData.data[{ {},i,{},{} }] = trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

print(sys.COLORS.red .. '==> preprocessing data: normalize all three channels locally')

--define Normalization neighborhood
neighborhood = image.gaussian1D(5) -- vedere come decidere questo parametro
local norm = nn.SpatialContrastiveNormalization(1,neighborhood,1):float() --1 input (canale y)

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = norm:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = norm:forward(testData.data[{ i,{c},{},{} }])
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red .. '==> verify statistics')

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

print(sys.COLORS.red .. '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local trSamples_y = trainData.data[{ {1,28},1 }]
   gfx.image(trSamples_y, {nrow=16, legend='Some training examples: Y channel'})
   local teSamples_y = testData.data[{ {1,8},1 }]
   gfx.image(teSamples_y, {nrow=16, legend='Some testing examples: Y channel', padding=2, zoom=4})
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
