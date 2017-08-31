function checkImage(img)
  os.execute('identify '..arg[1].. ' |  cut -f3 -d" " > tmp ')
  local file = io.open('tmp',"r")
  local size = file:read("*all")
  if size ~= '32x32' then 
   -- print('image scaling...')
    i = image.scale(img,32,32)
    return i
  end 
   
   return img
 
end

require 'image'
require 'nn'

img = image.load(arg[1])
net = torch.load(arg[2])

newImg = checkImage(img)
time = sys.clock() 

output = net:forward(newImg)


confidence, prediction = output:float():sort()
confidence = confidence[confidence:size(1)]
prediction = prediction[prediction:size(1)]

time = sys.clock() - time


classes = {'airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

prediction = classes[prediction]



print('time to classify the sample: '..math.floor((time*1000))..' ms')
print ('predizione = '..prediction)
