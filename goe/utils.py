import numpy as np

# Foolbox
from foolbox import PyTorchModel

def get_PyTorchModel(model, bounds, mean, std):
    # Generate preprocessing dict for PyTorchModel
    assert np.ndim(mean) == np.ndim(std)
    axis = -3 if np.ndim(mean)==1 else None # Prevent PyTorchModel error
    preprocessing = dict(mean=mean, std=std, axis=axis)

    return PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

class SmoothedPyTorchModel(PyTorchModel):
    pass