import numpy as np
import torch
from torchvision import transforms

# Foolbox
from foolbox import PyTorchModel

class UniversalModel(torch.nn.Module):
    def __init__(self,model,mean=0,std=1, bounds=(0,1)):
        self.mean = mean
        self.std = std
        self.norm = transforms.Normalize(mean, std)
        self.model = model
        axis = -3 if np.ndim(mean)==1 else None # Prevent PyTorchModel error
        self.fmodel = PyTorchModel(model, bounds=bounds, preprocessing=dict(mean=mean, std=std, axis=axis))
    def call(self,inputs):
        return self.model(self.norm(inputs))
    def fcall(self,inputs):
        return self.model(inputs)

def get_PyTorchModel(model, bounds, mean, std):
    # Generate preprocessing dict for PyTorchModel
    assert np.ndim(mean) == np.ndim(std)
    axis = -3 if np.ndim(mean)==1 else None # Prevent PyTorchModel error
    preprocessing = dict(mean=mean, std=std, axis=axis)

    return PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

class SmoothedPyTorchModel(PyTorchModel):
    pass
