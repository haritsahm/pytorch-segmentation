import torch
from NetModel import NetModel
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

# parameter = [Layer, Filter, Multiplier, Stride]
model = NetModel(input_channel=3, output_channel=2, param=[4, 4, 1.25, 2]).to(device)

# print(model)
summary(model, (3, 640, 480))

