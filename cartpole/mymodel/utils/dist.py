import torch.distributions as td
import torch
# from torchrl.modules import TruncatedNormal
class CustomTruncatedNormal(td.Normal):
    def __init__(self, loc, scale, low, high):
        super().__init__(loc, scale)
        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        sample = super().sample(*args, **kwargs)
        return torch.clamp(sample, self.low, self.high)

    def log_prob(self, value):
        log_prob = super().log_prob(value)
        log_Z = torch.log(self.cdf(self.high) - self.cdf(self.low))
        return log_prob - log_Z

class TruncNormalDist():
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.dist = torch.distributions.Independent(CustomTruncatedNormal(loc, scale, low, high), 1)
        self._clip = clip
        self._mult = mult

    def sample(self, *args, **kwargs):
        event = self.dist.sample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event

    @property
    def mode(self):
        return torch.clamp(self.loc, self.low, self.high)
    
    def normal_std(self):
        return self.scale.mean()
    
    def entropy(self):
        return torch.sum(torch.zeros_like(self.scale), -1)
    
    def log_prob(self, action):
        return torch.sum(torch.zeros_like(self.scale), -1)