import numpy as np

# Torch
import torch
from torchvision.transforms import Normalize

# Custom module
from goe.utils import get_PyTorchModel

def accuracy(model, dataloader, device, mean=0, std=1):
    model.eval()
    model = model.to(device)
    norm = Normalize(mean, std) if (mean, std)!=(0,1) else lambda x: x

    running_corrects = 0 # Count of correctly classified images
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        preds = model(norm(images)).argmax(1) # Get model's predictions
        running_corrects += torch.sum(preds == labels.data)
    return running_corrects.item() / len(dataloader.dataset)


def attack_successrate(model, dataloader, device, attack, mean=0, std=1,
                       model_bounds=None):
    """
    attack is a foolbox attack
        (fmodel, images, label) -> attack(fmodel, images, label)
    that does not accept kwargs. In order to still use them, wrap them in a
    function, as follows (example PGD-7):

    base_attack = LinfPGD(abs_stepsize=2/255, steps=7, random_start=True)
    attack_kwargs = {"epsilons": 8/255}

    def PGD7(fmodel, images, labels):
        return base_attack(fmodel, images, labels, **attack_kwargs)
    """
    model.eval()
    model = model.to(device)

    if model_bounds is None:
        print("No model_bounds provided. Using default: (0,1).")
        model_bounds = (0,1)
    fmodel = get_PyTorchModel(model, model_bounds, mean, std)

    running_successes = 0 # Count of imperceptible adversarial examples
    running_num_images = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        _, _, is_adv = attack(fmodel, images, labels)

        running_successes += is_adv.sum().item()
        running_num_images += len(labels)
        print(f"\r{running_successes = } / {running_num_images}",end="")
    print()
    assert len(dataloader.dataset) == running_num_images # Sanity check
    return running_successes / len(dataloader.dataset)

def L2_imperceptible(perturbation_tensor, epsilon):
    #TODO: Handle epsilon=None
    return (perturbation_tensor).norm(2, dim=[1,2,3]) <= epsilon

def transferattack_successrate(model, surrogate,  dataloader, device,attack,
                               mean=0, std=1, model_bounds=None,
                               check_imperceptible=None):

    if check_imperceptible is None:
        check_imperceptible = L2_imperceptible

    model.eval()
    surrogate.eval()
    model = model.to(device)
    surrogate = surrogate.to(device)

    if model_bounds is None:
        print("No model_bounds provided. Using default: (0,1).")
        model_bounds = (0,1)

    # fmodel is used for to handle preprocessing
    fmodel = get_PyTorchModel(model, model_bounds, mean, std)
    fsurrogate = get_PyTorchModel(surrogate, model_bounds, mean, std)

    running_successes = 0 # Count of imperceptible adversarial examples
    running_num_images = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        # Compute adversarial examples on surrogate
        _, advs, _ = attack(fsurrogate, images, labels)

        # Prediction of defender model and check if imperceptible
        epsilon = attack.eps # Use our attacks, not native Foolbox
        is_adv = fmodel(advs).argmax(1) != labels.data
        is_imperceptible = check_imperceptible(images-advs, epsilon)

        running_successes += torch.sum(is_adv & is_imperceptible).item()
        running_num_images += len(labels)
        print(f"\r{running_successes = } / {running_num_images}",end="")
    print()
    assert len(dataloader.dataset) == running_num_images # Sanity check
    return running_successes / len(dataloader.dataset)
