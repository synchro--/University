from mod_decomposer import pytorch_cp_layer_decomposition
from models.custom_models import *
from torchsummary import summary

device = 'cuda'
net = LenetZhang()
net = net.cpu()
batch_img = torch.randn(32, 3, 32, 32)
out = net(batch_img)

layer = net.conv3
net_cmp = pytorch_cp_layer_decomposition(layer)
#print(summary(net_cmp.to(device), (3, 32, 32)))

net.conv3 = net_cmp
print(summary(net.to(device), (3, 32, 32)))

