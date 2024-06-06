
import torch.nn as nn
import torch

m = nn.ConstantPad3d(3, 3.5)
test = torch.randn(16, 3, 10, 20, 30)
## first two dims are ignored because correspond to batch and channel
output = m(test)
print(output.size())
 # using different paddings for different sides
m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
#padding applied inversely to the order of dimensions of input tensor
output = m(test)
print(output.size())