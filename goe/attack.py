import warnings
import sys
from time import time
import numpy as np

# Torch
import torch

# Foolbox
import foolbox.attacks as fa


"""
Here we (re-)define attacks similar to foolbox.attacks.
Unlike foolbox.attacks, we pass the epsilon bound when we instantiate the
attack and not for __call__.

Advantage:
    No need to pass epsilon for each call. Results in cleaner code when
    defining adversarial training and repr also shows the specified epsilon.

Disadvantage:
    Lost flexibility.
"""

class PGD7(fa.LinfPGD):
    """
    PGD-7 by Madry et al. used for adversarial training in
    https://arxiv.org/abs/1706.06083
    """
    def __init__(self, epsilons):
        self.eps = epsilons
        super().__init__(abs_stepsize=self.eps/4, steps=7, random_start=True)

    def __call__(self, model, inputs, criterion):
        return super().__call__(model, inputs, criterion, epsilons=self.eps)


class L2CarliniWagnerAttack(fa.L2CarliniWagnerAttack):

    def __init__(self, epsilons, **kwargs):
        self.eps = epsilons
        super().__init__(**kwargs)

    def __call__(self, model, inputs, criterion):
        return super().__call__(model, inputs, criterion, epsilons=self.eps)

class InitBlackboxAttack:
    """
    Mixin class; Generate starting points before attacking.

    Apply attack to all inputs, where the init_BlackboxAttack was successful.
    The following construction ensures that inputs.size == advs.size
    """

    def __init__(self, epsilons, **kwargs):
        self.eps = epsilons
        super().__init__(**kwargs)
        # Same init_attack used by BoundaryAttack and HopSkipJump
        # Keep this as last item
        self.init_attack2 = fa.LinearSearchBlendedUniformNoiseAttack(steps=50)

    def __call__(self, model, inputs, criterion):

        with warnings.catch_warnings():
            # Filter warnings of failed attacks
            warnings.simplefilter('ignore', UserWarning)
            advs, clipped_advs, is_adv = self.init_attack2(
                model,
                inputs,
                criterion,
                epsilons=None,
            )

        advs[is_adv], clipped_advs[is_adv], is_adv = super().__call__(
            model,
            inputs[is_adv],
            criterion[is_adv],
            epsilons=self.eps,
            starting_points = clipped_advs[is_adv],
        )

        # For debugging
        if advs.isnan().any() or clipped_advs.isnan().any():
            raise ValueError(f"""At least one entry of the generated adversarial
                             examples is NaN.\n eps={self.eps}""")

        return advs, clipped_advs, is_adv


class BoundaryAttack(InitBlackboxAttack, fa.BoundaryAttack):
    pass

class HopSkipJumpAttack(InitBlackboxAttack, fa.HopSkipJumpAttack):
    pass

class L2UniversalAdversarialPerturbation(fa.L2DeepFoolAttack):
    """
    Calculates or loads universal adversarial perturbation
    https://arxiv.org/pdf/1610.08401.pdf

    If load_path is provided, then the universal adversarial perturbation is
    loaded from the specified npz-file. (Fields: "uap", "init_time")

    Otherwise it has to be calculated first with `calculate_perturbation`.
    """

    def __init__(self, epsilons, load_path=None, device=None):
        self.eps = epsilons
        self.uap = torch.zeros((3,32,32)).to(device)
        self.init_time = 0
        self.get_device(device)
        self.load_uap(load_path)


    def get_device(self, device):
        if device is None:
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"Default: UAP is loaded to {device_str}")
            self.device = torch.device(device_str)
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def load_uap(self, path):
        """
        path is None or should point to a npz file with fields
            "uap" (The universal adversarial perturbation as a numpy array)
            "init_time" (The time spent to compute the UAP)
        """
        if path is None:
            print("No UAP provided")
            return

        npzfile = np.load(path)
        self.uap = torch.from_numpy(npzfile["uap"]).to(self.device)
        self.init_time = npzfile["init_time"]

    def apply_perturbation(self, inputs):
        advs = inputs + self.uap.repeat_interleave(repeats=len(inputs), dim=0)
        clipped_advs = torch.clamp(advs, 0, 1) # TODO: Implement for other bounds
        return advs, clipped_advs

    def __call__(self, model, inputs, criterion):
        advs, clipped_advs = self.apply_perturbation(inputs)
        is_adv = model(advs).argmax(dim=1) != criterion
        return advs, clipped_advs, is_adv

    #########################################################
    # The following functions are needed to calculate the UAP
    #########################################################

    def L2projection(self, x):
        # Projects inside the L2 ball, not necessarily onto the L2 sphere.
        # The norm of the result can be smaller than epsilon
        norm = torch.linalg.vector_norm(x).item()
        return x * min(1, self.eps/norm)

    def calculate_foolingrate(self, model, dataloader):
        fooling_rate = 0
        ASR = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            _, advs = self.apply_perturbation(inputs)
            preds = model(inputs).argmax(dim=1)
            adv_preds = model(advs).argmax(dim=1)

            fooling_rate += torch.sum(preds != adv_preds).item()
            ASR += torch.sum(adv_preds!=labels).item()
        fooling_rate /= len(dataloader.dataset)
        ASR /= len(dataloader.dataset)
        print("Fooling rate:", fooling_rate)
        return fooling_rate

    def calculate_perturbation(self, model, trainloader, valloader, save_path=None,
                               save_best=True, df_steps=10, min_fooling_rate=0.8,
                               uap_maxiter=10):

        if save_best:
            print("save_best enabled: Early termination is not based on",
                  "min_fooling_rate. Instead we stop if and only if the",
                  "fooling_rate does not improve for a full epoch.")

        assert model.bounds == (0,1) #TODO: Implement for other bounds
        super().__init__(steps=df_steps)

        best_fooling_rate = 0
        begin = time()
        for i in range(uap_maxiter):
            print("UAP Iteration:",i,"Data loader size:",f"{len(trainloader.dataset)}")
            fooling_rate_improved = False
            for k, (inputs, labels) in enumerate(trainloader):
                print()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                for input_, label in zip(inputs[:,None,:],labels[:,None]):

                    _, uap_input = self.apply_perturbation(input_)
                    uap_prediction = model(uap_input).argmax(dim=1)
                    # If the classifier is not fooled...
                    if uap_prediction == label:

                        # Deepfool attack
                        adv, _, is_adv = super().__call__(
                            model, uap_input, uap_prediction, epsilons=None,
                        )
                        if is_adv:
                            self.uap = self.L2projection(adv-input_)
                print()
                print("Time:", time()-begin)
                fooling_rate = self.calculate_foolingrate(model, valloader)
                if best_fooling_rate < fooling_rate:
                    fooling_rate_improved = True
                    print("New best fooling_rate:")
                    print(f"{best_fooling_rate=},{fooling_rate=}")
                    best_fooling_rate = fooling_rate
                    if save_path is not None and save_best:
                        self.save_uap(save_path, time()-begin, self.uap)

                # Early termination criterions.
                # First one is based on min_fooling rate. Second checks if
                # fooling_rate has not been improved for an epoch
                if fooling_rate >= min_fooling_rate and not save_best: break
            if save_best and not fooling_rate_improved: break
            else: print("finished iteration:",i,"- fooling rate increased")
        else:
            print(f"Computation did not terminate early.")

        if save_path is not None and not save_best:
            self.init_time = time() - begin
            save_uap(save_path, self.init_time, self.uap)

    def save_uap(self, save_path, init_time, uap):
        uap = uap.cpu().detach().numpy()
        np.savez(save_path, init_time=init_time, uap=uap)

    def __repr__(self):
        return f'UAP(eps={self.eps})'
