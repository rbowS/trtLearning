import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(1,1,3,padding=1)
        self.relu = nn.ReLU()
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


model = Model()
dummpy = torch.zeros(1,1,3,3)
torch.onnx.export(
    model,
    # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
    (dummpy,),
    # 储存的文件路径
    "demo.onnx",
    verbose=False,
    # 为输入和输出节点指定名称，方便后面查看或者操作
    input_names=["image"],
    output_names=["output"],
    # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
    opset_version=11,
    # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
    dynamic_axes={
        "image":{0:"batch", 2:"height", 3:"width"},
        "output":{0:"batch", 2:"height", 3:"width"},
    }
)

print("Done")