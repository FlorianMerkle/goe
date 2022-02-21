import warnings
import sys
from time import time

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

    Tip: Repr output can be more pretty when using fraction.Fraction(x,y) as
    epsilons
    """
    def __init__(self, epsilons):
        self.eps = epsilons
        super().__init__(abs_stepsize=float(self.eps/4), steps=7, random_start=True)

    def __call__(self, model, inputs, criterion):
        return super().__call__(model, inputs, criterion, epsilons=float(self.eps))


class L2CarliniWagnerAttack(fa.L2CarliniWagnerAttack):

    def __init__(self, epsilons, **kwargs):
        self.eps = epsilons
        super().__init__(**kwargs)

    def __call__(self, model, inputs, criterion):
        return super().__call__(model, inputs, criterion, epsilons=self.eps)


class init_BoundaryAttack(fa.LinearSearchBlendedUniformNoiseAttack):
    """
    The same init_attack is used by the default Foolbox BoundaryAttack.
    However, this attack can fail, causing errors for the boundary attack
    We remove all unsuccessful adversarial examples to prevent these errors.
    """
    def __init__(self):
        super().__init__(steps=50)

    def __call__(self, model, inputs, criterion):
        with warnings.catch_warnings():
            # Filter warnings of failed attacks
            warnings.simplefilter('ignore', UserWarning)
            _, advs, is_adv = super().__call__(
                model, inputs, criterion, epsilons=None,
            )
        return inputs[is_adv], criterion[is_adv], advs[is_adv]


class BoundaryAttack(fa.BoundaryAttack):
    """
    The same init_attack is used by the default Foolbox BoundaryAttack.
    However, this attack can fail, causing errors for the boundary attack
    We remove all unsuccessful adversarial examples to prevent these errors.
    """
    def __init__(self, epsilons, **kwargs):
        self.eps = epsilons
        self.init_attack_ = init_BoundaryAttack()
        super().__init__(**kwargs)

    def __call__(self, model, inputs, criterion):
        inputs, criterion, advs = self.init_attack_(model, inputs, criterion)
        return super().__call__(
            model, inputs, criterion,
            epsilons=self.eps, starting_points=advs
        )

class L2UniversalAdversarialPerturbation(fa.L2DeepFoolAttack):
    """
    Universal adversarial perturbations based on paper
    https://arxiv.org/pdf/1610.08401.pdf

    Minimal example:
        epilon = 10,
        trainloader = ... (dataloader of training set),
        valloader = ... (dataloader of validation set)
        device = torch.device (gpu or cpu)
        model = PyTorchModel(...)

        attack = L2UniversalAdversarialPerturbation(epsilon)
        attack.calculate_perturbation(model, trainloader, valloader, device)

        Then attack(model, inputs, criterions) works as usual
    """
    def __init__(self, epsilons, df_steps=10, min_fooling_rate=0.8, uap_maxiter=10):
        self.uap = 0 # Universal adversarial pertubation
        self.eps = epsilons
        self.min_fooling_rate = min_fooling_rate
        self.uap_maxiter = uap_maxiter # Max number of times we iterate through dataset
        super().__init__(steps=df_steps)

    def apply_perturbation(self, inputs):
        advs = inputs + self.uap.repeat_interleave(repeats=len(inputs), dim=0)
        clipped_advs = torch.clamp(advs, 0, 1) # TODO: Implement for other bounds
        return advs, clipped_advs

    def L2projection(self, x):
        # Projects inside the L2 ball, not necessarily onto the L2 sphere.
        # The norm of the result can be smaller than self.eps
        norm = torch.linalg.vector_norm(x).item()
        return x * min(1, self.epsilons/norm)

    def calculate_foolingrate(self, model, dataloader, device):
        fooling_rate = 0
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _, advs = self.apply_perturbation(inputs)
            preds = model(inputs).argmax(dim=1)
            adv_preds = model(advs).argmax(dim=1)

            fooling_rate += torch.sum(preds != adv_preds).item()
        fooling_rate /= len(dataloader.dataset)
        print(f"Fooling rate = {fooling_rate}")
        return fooling_rate

    def calculate_perturbation(self, model, trainloader, valloader, device):
        assert model.bounds == (0,1) #TODO: Implement for other bounds

        for k in range(self.uap_maxiter):
            running_fooling_rate = 0
            begin = time()
            print("Data loader size:",f"{len(trainloader.dataset)}")
            for k, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                for input_, label in zip(inputs[:,None,:],labels[:,None]):
                    # Deepfool attack
                    adv, clipped_adv, is_adv = super().__call__(
                        model, input_+self.uap, label, epsilons=self.eps
                    )

                    if model(input_).argmax() != is_adv:
                        # No adversarial example found, don't update universal
                        # perturbation
                        continue
                    running_fooling_rate += 1
                    perturbation = (adv - input_) + self.uap
                    self.uap = self.L2projection(perturbation)
                print(f"\r{running_fooling_rate}","/",(k+1)*len(labels),end="")
                sys.stdout.flush()
            print()
            print(time()-begin)
            fooling_rate = self.calculate_foolingrate(model, valloader, device)
            if fooling_rate < self.min_fooling_rate:
                break

    def __call__(self, model, inputs, criterion):
        advs, clipped_advs = self.apply_perturbation(inputs)
        is_adv = model(advs).argmax(dim=1) != criterion
        return advs, clipped_advs, is_adv

    def __repr__(self) -> str:
        args = ", ".join(f"{k.strip('_')}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"
