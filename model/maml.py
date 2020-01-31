import copy
import torch
from torch.autograd import grad
from model import BaseLearner


class MAML(BaseLearner):
    def __init__(self, model, lr, first_order=False, allow_unused=False):
        super(MAML, self).__init__()
        # print('Inner train model structure\n %s' % str(model))
        self.module = model  # Module to be wrapped
        self.lr = lr  # Fast adaptation learning rate
        self.first_order = first_order  # Whether to use the first-order approximation of MAML
        self.allow_unused = allow_unused  # Whether to allow differentiation of unused parameters

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, first_order=None, allow_unused=None):
        # Updates the clone parameters in place using the MAML update
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        second_order = not first_order
        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order,
                         allow_unused=allow_unused)
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None, allow_unused=None):
        # Returns a MAML-wrapped copy of the module whose parameters and buffers are cloned from original module
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused)


def maml_update(model, lr, grads=None):
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            print('WARNING:maml_update(): Parameters and gradients have different length. (%d vs %d)'
                  % (len(params), len(grads)))
        for p, g in zip(params, grads):
            p.grad = g

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = maml_update(model._modules[module_key], lr=lr, grads=None)
    return model


def clone_parameters(param_list):
    # return a cloned parameter list
    return [p.clone() for p in param_list]


def clone_module(module):
    # Creates a copy of a module, whose parameters/buffers/submodules are created using PyTorch's torch.clone().
    clone = copy.deepcopy(module)

    # First, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                cloned = module._parameters[param_key].clone()
                clone._parameters[param_key] = cloned

    # Second, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                clone._buffers[buffer_key] = module._buffers[buffer_key].clone()

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(module._modules[module_key])
    return clone


def detach_module(module):
    # Detaches all parameters/buffers of a previously cloned module from its computational graph
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key])


