import torch
import math

class StraightThroughEstimator(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
    
def straight_through_estimator():
    def inner(x):
        return StraightThroughEstimator.apply(x)
    
    return inner

class Triangular(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, threshold=1):
        ctx.save_for_backward(input_)
        ctx.threshold = threshold
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ctx.threshold
        grad[input_ >= 0] = -grad[input_ >= 0]
        return grad, None
    
    
def triangular():
    def inner(x):
        return Triangular.apply(x)
    return inner


class FastSigmoid(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def fast_sigmoid(slope=25):
    slope = slope
    
    def inner(x):
        return FastSigmoid.apply(x, slope)
    
    return inner

class ATan(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, alpha=2.0):
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = ((ctx.alpha / 2 ) / (1+ (math.pi / 2 * ctx.alpha * input_).pow_(2)) * grad_input)
        return grad, None
    
    
def atan(alpha=2.0):
    alpha = alpha
    
    def inner(x):
        return ATan.apply(x, alpha)
    
    return inner

@staticmethod
class Heaviside(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (out, ) = ctx.saved_tensors
        grad = grad_output * out
        return grad
    
    
def heaviside():
    
    def inner(x):
        return Heaviside.apply(x)
    
    return inner

class Sigmoid(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input * ctx.slope * torch.exp(-ctx.slope * input_) / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        return grad, None
    
def sigmoid(slope=25):
    
    slope = slope
    
    def inner(x):
        return Sigmoid.apply(x)
    
    return inner

class SpikeRateEscape(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, beta=1, slope=25):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        ctx.slope = slope
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input * ctx.slope * torch.exp(-ctx.beta * torch.abs(input_ - 1))
        )
        return grad, None, None
    
def spike_rate_escape(beta = 1, slope = 25):
    
    beta = beta
    slope = slope
    
    def inner(x):
        return SpikeRateEscape.apply(x, beta, slope)
    
    return inner

class StochasticSpikeOperator(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, mean=0, variance=0.2):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.mean = mean
        ctx.variance = variance
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * out + (grad_input * (~out.bool()).float()) * (
            (torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance
        )
        return grad, None, None
    
def SSO(mean=0, variance=0.2):
    mean = mean
    variance = variance
    
    def inner(x):
        return StochasticSpikeOperator.apply(x, mean, variance)
    
    return inner

class LeakySpikeOperator(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, slope=0.1):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        ctx.slope = slope
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (out, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input * out + (~out.bool()).float() * ctx.slope * grad_input
        )
        return grad
    
def LSO(slope=0.1):
    slope = slope
    
    def inner(x):
        return StochasticSpikeOperator.apply(x, slope)
    
    return inner

class SparseFastSigmoid(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, slope=25, B=1):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.B = B
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2 * (input_ > ctx.B).float()
        )
        return grad, None, None
    
def SFS(slope=25, B=1):
    
    slope = slope
    B = B
    
    def inner(x):
        return SparseFastSigmoid.apply(x, slope, B)
    
    return inner

class FractionalOrderGradient(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, alpha=0.9):
        ctx.save_for_backward(input_)
        ctx.gamma_value = math.gamma(1 - alpha)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        (input_, ) = ctx.saved_tensors
        grad = (
            1 / (ctx.gamma_value * torch.pow(input_, ctx.alpha))
        )
        return grad
    
def FOG(alpha=0.9):
    
    alpha = alpha
    
    def inner(x):
        return FractionalOrderGradient.apply(x, alpha)
    
    return inner

