from mod_decomposer import pytorch_cp_layer_decomposition
from models.custom_models import *
from torchsummary import summary

device = 'cuda'
net = torch.load('prova.pth')
net = net.cpu()

layer = net.conv4
layer_cmp = pytorch_cp_layer_decomposition(layer)
print(summary(net.to(device), (3, 32, 32)))

net.conv4 = layer_cmp
print(summary(net.to(device), (3, 32, 32)))

torch.save(net, 'decomposed.pth')
