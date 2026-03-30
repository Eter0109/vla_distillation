import torch

class FeatureHook:
    def __init__(self, module):
        self.output = None
        self.hook = module.register_forward_hook(self.fn)

    def fn(self, module, inp, out):
        self.output = out.detach().clone()

    def close(self):
        self.hook.remove()