import numpy as np
import pandas as pd
from time import time

# Torch
import torch
from torchvision.transforms import Normalize

# Plotting
import matplotlib.pyplot as plt
from livelossplot import PlotLosses

# Custom module
from codebase.src.utils import get_PyTorchModel

# TODO: Replace PlotLosses with pandas + matplotlib

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs,
                save_name=None, scheduler=None, mean=0, std=1,
                attack=None, model_bounds=None):

    # Preperation
    begin = time()
    model = model.to(device) # Moves/casts the parameters and buffers to device
    norm = Normalize(mean, std) if (mean, std)!=(0,1) else lambda x: x
    best_val_acc = 0
    plt.ion()

    ## Initialize fmodel for adversarial training (if necessary)
    if attack is None:
        print("Standard training")
    else:
        print("Adversarial training")
        if model_bounds is None:
            print("No model_bounds provided. Using default: (0,1).")
            model_bounds = (0,1)
        model.eval() # Prevent UserWarning, does not affect training below.
        fmodel = get_PyTorchModel(model, model_bounds, mean, std)
    
    if attack != None: columns = ['acc', 'loss', 'rob_acc', 'rob_loss', 'val_acc', 'val_loss']
    if attack == None: columns = ['acc', 'loss', 'val_acc', 'val_loss']
    print('    '.join(columns))
    df = pd.DataFrame(columns=columns)
    # Begin training
    for epoch in range(1, num_epochs+1):
        logs = {}
        pd_logs = []
        for phase in ['train', 'validation']: # First train, then validate
            # Switch between training and test eval mode depending on phase.
            # If the model is adversarially trained, then model.train() is set
            # later on
            model.train() if phase == 'train' and attack is None else model.eval()

            running_loss = 0.0
            running_corrects = 0
            if attack is not None:
                adv_running_loss = 0.0
                adv_running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                # Compute gradients and update weights
                outputs = model(norm(images))
                loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1) # Get model's predictions
                running_loss += loss.detach() * images.size(0) # Convert batch mean loss to sum of losses
                running_corrects += torch.sum(preds == labels.data) # add number of correct predictions to total

                if phase == "train" and attack is not None:
                    # Adversarially perturb images before computing gradients
                    _, images, _ = attack(fmodel, images, labels)

                    model.train()
                    outputs = model(norm(images))
                    loss = criterion(outputs, labels)

                    preds = torch.argmax(outputs, dim=1) # Get model's predictions
                    adv_running_loss += loss.detach() * images.size(0) # Convert batch mean loss to sum of losses
                    adv_running_corrects += torch.sum(preds == labels.data) # add num of correct predictions to total

                if phase == "train":
                    optimizer.zero_grad() # Set previously gradients to 0
                    loss.backward() # Calculate gradients
                    optimizer.step() # Update weights

            num_samples = len(dataloaders[phase].dataset)
            epoch_loss = running_loss.item() / num_samples # mean loss of epoch
            epoch_acc = running_corrects.item() / num_samples
            pd_logs.extend((epoch_acc, epoch_loss))
            if phase == "train" and attack is not None:
                adv_epoch_loss = adv_running_loss.item() / num_samples
                adv_epoch_acc = adv_running_corrects.item() / num_samples
                pd_logs.extend((adv_epoch_acc, adv_epoch_loss))            
            # Logging
            prefix = 'val_' if phase == 'validation' else 'train_'
            logs[prefix + 'loss'] = epoch_loss
            logs[prefix + 'acc'] = epoch_acc
            if attack is not None and phase == 'train':
                logs['rob_' + 'loss'] = adv_epoch_loss
                logs['rob_' + 'acc'] = adv_epoch_acc

            # Save best model if 'save_name' was provided.
            if phase == 'validation' and epoch_acc>best_val_acc:
                best_val_acc = epoch_acc
                if save_name is not None:
                    accstr = str(np.round(epoch_acc,6))

                    accstr += (6-len(accstr))*'0'
                    torch.save(model, save_name + ".pt")
                    with open(save_name + ".txt", "w") as f:
                        f.write(f"{save_name}\n")
                        f.write(f"Best epoch={epoch+1}, valacc={accstr}\n")
                        f.write(f"Computation time for {epoch+1} epochs {time()-begin}\n")
                    print("New best validation accuracy - Model saved")
        if scheduler is not None:
            scheduler.step()
        print('  '.join(list(map(lambda x: str(round(x,3)),pd_logs))))
    
        s_logs = pd.Series(pd_logs, index=columns)
        df = df.append(s_logs, ignore_index=True)
        plt.close()
        if attack != None: df.plot(secondary_y=['loss','rob_loss','val_loss'] , style=['-',':','-',':','-',':'])
        if attack == None: df.plot(secondary_y=['loss','val_loss'] , style=['-',':','-',':'])
        df.to_csv(f'{save_name}.csv')
        plt.savefig(f'{save_name}.png')
        plt.pause(1e-10)

    if save_name is not None:
        torch.save(model, save_name + ".pt")
        with open(save_name + ".txt", "w") as f:
            f.write(f"{save_name}\n")
            f.write(f"Best epoch={epoch+1}, valacc={accstr}, {save_name}")
            f.write(f"Computation time for {num_epochs} epochs {time()-begin}\n")
        print("Model saved")
        print()
