from mod_decomposer import *
from models.custom_models import *
from torchsummary import summary

device = 'cuda'
net = torch.load('prova.pth')
net = net.cpu()

layer = net.conv2
layer_cmp = pytorch_tucker_layer_decomposition(layer)
print(summary(net.to(device), (3, 32, 32)))

net.conv2 = layer_cmp
print(summary(net.to(device), (3, 32, 32)))

torch.save(net, 'decomposed.pth')
