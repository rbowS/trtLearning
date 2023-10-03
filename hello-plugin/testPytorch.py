import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import os

class MYSELUImpl(torch.autograd.Function):

    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, x, p):
        print("==================================call symbolic")
        return g.op("MYSELU", x, p, 
            g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
            attr1_s="这是字符串属性", 
            attr2_i=[1, 2, 3], 
            attr3_f=222
        )

    @staticmethod
    def forward(ctx, x, p):
        return x * 1 / (1 + torch.exp(-x))


class MYSELU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.arange(n).float())

    def forward(self, x):
        return MYSELUImpl.apply(x, self.param)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.myselu = MYSELU(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.myselu(x)
        return x
    

model = Model().eval()
input = torch.tensor([
    # batch 0
    [
        [1,   1,   1],
        [1,   1,   1],
        [1,   1,   1],
    ],
        # batch 1
    [
        [-1,   1,   1],
        [1,   0,   1],
        [1,   1,   -1]
    ]
], dtype=torch.float32).view(2, 1, 3, 3)

output = model(input)
print(f"inference output = \n{output}")