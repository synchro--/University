# PyTorch Notes 

## Freeze params: 

```python 
lt=8
cntr=0

for child in model.children():
cntr+=1

if cntr < lt:
	print child
	for param in child.parameters():
		param.requires_grad = False

optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

# Getter & Setter for layers 
model.fc1.weight.data.numpy() #to get numpy array of weights 
model.fc1.weight # get weights
model.fc1.weight.data = random_tensor # set weigths!
model.fc1.weight.size() # or .shape return torch.size()
model.conv.weight.data.numpy().shape 
# returns a shape like this: 
(32, 16, 3, 3) --> (OUT, IN, d_h, d_w)

num_ftrs = model_ft.fc.in_features # number of input features 
```

# Feature extractor 

```python

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

## Dataloadaer 

```python
# Cifar 
In [65]: len(trainloader.dataset)
Out[65]: 50000

In [66]: trainloader.batch_size
Out[66]: 32

In [65]: it = iter(trainloader)

In [63]: it.next()[0].shape 
Out[63]: torch.Size([32, 3, 32, 32])

In [64]: it.next()[1].shape 
Out[64]: torch.Size([32])
```

# AWK Sum line
```sh 
 # Sum every value at beginning of line with 75K 
 awk -F',' '{OFS=","}{print $1=$1+75000,$2,$3}' 
```