from pytorch_utils import *  

# Test
import torchvision.models as models
model = models.alexnet()
print(model)
print('\n\n#### WITH torch_summarize: ####')
print(torch_summarize(model))

# # Output
# AlexNet (
#   (features): Sequential (
#     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)), weights=((64, 3, 11, 11), (64,)), parameters=23296
#     (1): ReLU (inplace), weights=(), parameters=0
#     (2): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1)), weights=(), parameters=0
#     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)), weights=((192, 64, 5, 5), (192,)), parameters=307392
#     (4): ReLU (inplace), weights=(), parameters=0
#     (5): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1)), weights=(), parameters=0
#     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), weights=((384, 192, 3, 3), (384,)), parameters=663936
#     (7): ReLU (inplace), weights=(), parameters=0
#     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), weights=((256, 384, 3, 3), (256,)), parameters=884992
#     (9): ReLU (inplace), weights=(), parameters=0
#     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), weights=((256, 256, 3, 3), (256,)), parameters=590080
#     (11): ReLU (inplace), weights=(), parameters=0
#     (12): MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1)), weights=(), parameters=0
#   ), weights=((64, 3, 11, 11), (64,), (192, 64, 5, 5), (192,), (384, 192, 3, 3), (384,), (256, 384, 3, 3), (256,), (256, 256, 3, 3), (256,)), parameters=2469696
#   (classifier): Sequential (
#     (0): Dropout (p = 0.5), weights=(), parameters=0
#     (1): Linear (9216 -> 4096), weights=((4096, 9216), (4096,)), parameters=37752832
#     (2): ReLU (inplace), weights=(), parameters=0
#     (3): Dropout (p = 0.5), weights=(), parameters=0
#     (4): Linear (4096 -> 4096), weights=((4096, 4096), (4096,)), parameters=16781312
#     (5): ReLU (inplace), weights=(), parameters=0
#     (6): Linear (4096 -> 1000), weights=((1000, 4096), (1000,)), parameters=4097000
#   ), weights=((4096, 9216), (4096,), (4096, 4096), (4096,), (1000, 4096), (1000,)), parameters=58631144
# )