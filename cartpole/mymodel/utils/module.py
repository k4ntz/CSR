import torch.nn as nn
from typing import Iterable
from ..models.drssm_cartpole import DRSSM

'''
def get_parameters(modules: Iterable[nn.Module], adapt_theta=False):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters
'''

def get_parameters(modules, adapt_theta=False):
    """
    Given a list of torch modules, returns a list of their parameters, optionally excluding theta_deter and theta_stoch.
    :param modules: iterable of modules
    :param adapt_theta: bool, when True, theta_deter and theta_stoch are not included in the parameters list
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        # Check if the module is an instance of DRSSM and adapt_theta is True
        if isinstance(module, DRSSM) and adapt_theta:
            # Loop through each parameter in the module
            for param in module.parameters():
                # Exclude theta_deter and theta_stoch by comparing objects directly
                if param is not module.theta_deter and param is not module.theta_stoch:
                    model_parameters.append(param)
        else:
            # If not excluding, add all parameters normally
            model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module], adapt_theta=False):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.adapt_theta = adapt_theta
        self.param_states = [p.requires_grad for p in get_parameters(self.modules, self.adapt_theta)]
        
    def __enter__(self):
        for param in get_parameters(self.modules, self.adapt_theta):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules, self.adapt_theta)):
            param.requires_grad = self.param_states[i]