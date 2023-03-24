import torch
import math

class StraightThroughEstimator(torch.autograd.Function):
    """
    Implements the straight-through estimator for a binary step function.
    """
    @staticmethod
    def forward(ctx, input_):
        """
        Computes the forward pass for the binary step function.

        Args:
            input_ (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with the same shape as input_, where values
            greater than 0 are set to 1 and values less than or equal to 0 are set
            to 0.
        """
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the backward pass for the binary step function.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input, which is the
            same as the gradient of the loss with respect to the output, but with the
            gradient of the binary step function (which is 0 almost everywhere) replaced
            with 1.
        """
        grad_input = grad_output.clone()
        return grad_input
    
    
def straight_through_estimator():
    """
    Returns a function that applies the straight-through estimator for a binary step function
    to its input.

    Returns:
        callable: A function that takes a tensor as input and returns a tensor with the same
        shape, where values greater than 0 are set to 1 and values less than or equal to 0 are
        set to 0.
    """
    def inner(x):
        return StraightThroughEstimator.apply(x)
    
    return inner

class Triangular(torch.autograd.Function):
    """
    Implements the triangular function, which is a binary step function with a slope
    of `threshold` at the origin and a slope of -`threshold` at `input_` = `threshold`.
    """
    @staticmethod
    def forward(ctx, input_, threshold=1):
        """
        Computes the forward pass for the triangular function.

        Args:
            input_ (torch.Tensor): Input tensor.
            threshold (float): Threshold value for the function. Default is 1.

        Returns:
            torch.Tensor: Output tensor with the same shape as input_, where values
            greater than 0 are set to 1 and values less than or equal to 0 are set
            to 0.
        """
        ctx.save_for_backward(input_)
        ctx.threshold = threshold
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the backward pass for the triangular function.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple of torch.Tensor: Gradient of the loss with respect to the input, which is the
            same as the gradient of the loss with respect to the output, but with the gradient
            of the triangular function replaced with `threshold` for values less than 0 and
            replaced with -`threshold` for values greater than or equal to `threshold`. The
            second element of the tuple is None, as there are no gradients with respect to the
            threshold.
        """
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ctx.threshold
        grad[input_ >= 0] = -grad[input_ >= 0]
        return grad, None
    
    
def triangular():
    """
    Creates a callable that applies the Triangular function to its input tensor.

    Returns:
        callable: A callable that applies the Triangular function to its input tensor.
    """
    def inner(x):
        return Triangular.apply(x)
    return inner


class FastSigmoid(torch.autograd.Function):
    """
    A faster approximation of the sigmoid function.

    Attributes:
        slope (float): The slope parameter of the FastSigmoid function.
    """
    @staticmethod
    def forward(ctx, input_, slope=25):
        """
        Applies the FastSigmoid function to the input tensor.

        Args:
            input_ (torch.Tensor): The input tensor.
            slope (float): The slope parameter of the FastSigmoid function.

        Returns:
            torch.Tensor: The output tensor of the FastSigmoid function.
        """
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradients of the FastSigmoid function with respect to its input.

        Args:
            ctx (torch.autograd.FunctionContext): The context object for the backward pass.
            grad_output (torch.Tensor): The gradient of the loss with respect to the output tensor.

        Returns:
            torch.Tensor: The gradient of the loss with respect to the input tensor.
        """
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def fast_sigmoid(slope=25):
    """
    Creates a callable that applies the FastSigmoid function to its input tensor.

    Args:
        slope (float): The slope parameter of the FastSigmoid function.

    Returns:
        callable: A callable that applies the FastSigmoid function to its input tensor.
    """
    slope = slope
    
    def inner(x):
        return FastSigmoid.apply(x, slope)
    
    return inner

class ATan(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, alpha=2.0):
        """
        Computes the forward pass of the ATan function.

        Args:
            input_ (Tensor): The input tensor.
            alpha (float): The scaling factor of the ATan function.

        Returns:
            Tensor: The output tensor after applying the ATan function.
        """
        ctx.save_for_backward(input_)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the backward pass of the ATan function.

        Args:
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, None]: The gradient of the input tensor and None for the scaling factor.
        """
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = ((ctx.alpha / 2 ) / (1+ (math.pi / 2 * ctx.alpha * input_).pow_(2)) * grad_input)
        return grad, None
    
    
def atan(alpha=2.0):
    """
    Creates and returns a closure that applies the ATan function with a specified scaling factor.

    Args:
        alpha (float): The scaling factor of the ATan function.

    Returns:
        Callable[[Tensor], Tensor]: A closure that takes an input tensor and applies the ATan function with the specified scaling factor.
    """
    alpha = alpha
    
    def inner(x):
        return ATan.apply(x, alpha)
    
    return inner

@staticmethod
class Heaviside(torch.autograd.Function):
    """
    The Heaviside function, which is used in the forward and backward pass of the network.

    Args:
        input_ (Tensor): The input tensor.

    Returns:
        Tensor: The output tensor after applying the Heaviside function to the input tensor.

    """
    @staticmethod
    def forward(ctx, input_):
        """
        Applies the Heaviside function to the input tensor.

        Args:
            input_ (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the Heaviside function to the input tensor.
        """
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the backward pass for the Heaviside function.

        Args:
            ctx (Context): The context object for backpropagation.
            grad_output (Tensor): The gradient of the output tensor with respect to some scalar value.

        Returns:
            Tensor: The gradient of the input tensor with respect to some scalar value.
        """
        (out, ) = ctx.saved_tensors
        grad = grad_output * out
        return grad
    
    
def heaviside():
    """
    Returns a closure function that applies the Heaviside autograd function to the input tensor.

    The Heaviside function returns a tensor of the same shape as input tensor `x`, with each element of the tensor being 0 or 1,
    depending on the sign of the corresponding element in `x`.

    Returns:
    - inner: a closure function that applies the Heaviside autograd function to the input tensor `x`.
    """
    def inner(x):
        """
        Applies the Heaviside autograd function to the input tensor `x`.

        Args:
        - x: the input tensor.

        Returns:
        - out: a tensor of the same shape as input tensor `x`, with each element of the tensor being 0 or 1,
               depending on the sign of the corresponding element in `x`.
        """
        return Heaviside.apply(x)
    
    return inner

class Sigmoid(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, slope=25):
        """
        Forward pass of the sigmoid function.
        
        Args:
        - input_ (torch.Tensor): The input tensor.
        - slope (float): The slope parameter of the sigmoid function.
        
        Returns:
        - torch.Tensor: The output tensor after applying the sigmoid function.
        """
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the sigmoid function.
        
        Args:
        - ctx (tuple): A tuple containing the context saved in the forward pass.
        - grad_output (torch.Tensor): The gradient of the loss with respect to the output of the sigmoid function.
        
        Returns:
        - tuple: A tuple containing the gradient of the loss with respect to the input of the sigmoid function,
        and None for the slope since it is a hyperparameter.
        """
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input * ctx.slope * torch.exp(-ctx.slope * input_) / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        return grad, None
    
def sigmoid(slope=25):
    """
    Create a callable instance of the Sigmoid function with a specific slope parameter.
    
    Args:
    - slope (float): The slope parameter of the sigmoid function.
    
    Returns:
    - callable: A callable instance of the Sigmoid function with the specified slope parameter.
    """
    slope = slope
    
    def inner(x):
        return Sigmoid.apply(x)
    
    return inner

class SpikeRateEscape(torch.autograd.Function):
    """
    The SpikeRateEscape function applies the Spike Rate Escape function to the input tensor. 
    This function is a non-linear activation function used in artificial neural networks for 
    spiking neurons. It enables neurons to emit spikes at a high rate even when their input 
    is high and persistent. 
    
    Args:
    - input_ (torch.Tensor): Input tensor of any shape.
    - beta (float): Parameter that controls the rate of escape of the neuron. Defaults to 1.
    - slope (float): Parameter that controls the slope of the activation function. Defaults to 25.

    Returns:
    - torch.Tensor: The output tensor after applying the Spike Rate Escape function element-wise to the input tensor.
    """
    @staticmethod
    def forward(ctx, input_, beta=1, slope=25):
        """
        Computes the forward pass of the SpikeRateEscape function.

        Args:
        - input_ (torch.Tensor): Input tensor of any shape.
        - beta (float): Parameter that controls the rate of escape of the neuron. Defaults to 1.
        - slope (float): Parameter that controls the slope of the activation function. Defaults to 25.

        Returns:
        - torch.Tensor: The output tensor after applying the Spike Rate Escape function element-wise to the input tensor.
        """
        ctx.save_for_backward(input_)
        ctx.beta = beta
        ctx.slope = slope
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the backward pass of the SpikeRateEscape function.

        Args:
        - ctx (torch.autograd.Function): Context object used to store information needed for the backward pass.
        - grad_output (torch.Tensor): Gradient tensor of any shape.

        Returns:
        tuple: A tuple containing three elements:
            - torch.Tensor: The gradient tensor with respect to the input tensor.
            - None: There is no gradient with respect to beta.
            - None: There is no gradient with respect to slope.
        """
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
        """
        Forward pass of the stochastic spike operator.
        
        Parameters:
        -----------
        input_ : torch.Tensor
            The input tensor to the spike function.
        mean : float, optional
            The mean value of the random distribution used for generating the stochastic spikes.
            Default is 0.
        variance : float, optional
            The variance value of the random distribution used for generating the stochastic spikes.
            Default is 0.2.
        
        Returns:
        --------
        out : torch.Tensor
            The output tensor with binarized values based on the random distribution and threshold.

        """
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.mean = mean
        ctx.variance = variance
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the stochastic spike operator.
        
        Parameters:
        -----------
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output of the spike function.
        
        Returns:
        --------
        grad : torch.Tensor
            The gradient of the loss with respect to the input of the spike function.
        None
        None
        """
        (input_, out) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * out + (grad_input * (~out.bool()).float()) * (
            (torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance
        )
        return grad, None, None
    
def SSO(mean=0, variance=0.2):
    """
    Wrapper of StochasticSpikeOperator
    """
    mean = mean
    variance = variance
    
    def inner(x):
        return StochasticSpikeOperator.apply(x, mean, variance)
    
    return inner

class LeakySpikeOperator(torch.autograd.Function):
    """
    Implements a Leaky Integrate-and-Fire (LIF) neuron using PyTorch's autograd functionality.
    
    Args:
    - input_ (tensor): Input signal.
    - slope (float): Slope of the leaky membrane potential. Default value is 0.1.
    
    Returns:
    - out (tensor): Binary tensor indicating whether the neuron has fired a spike or not.
    
    Backward args:
    - grad_output (tensor): Gradient of the loss with respect to the output.
    
    Backward returns:
    - grad (tensor): Gradient of the loss with respect to the input.
    """
    @staticmethod
    def forward(ctx, input_, slope=0.1):
        """
        Computes the forward pass of the LIF neuron.
        
        Args:
        - input_ (tensor): Input signal.
        - slope (float): Slope of the leaky membrane potential. Default value is 0.1.
        
        Returns:
        - out (tensor): Binary tensor indicating whether the neuron has fired a spike or not.
        """
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        ctx.slope = slope
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the backward pass of the LIF neuron.
        
        Args:
        - grad_output (tensor): Gradient of the loss with respect to the output.
        
        Returns:
        - grad (tensor): Gradient of the loss with respect to the input.
        """
        (out, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input * out + (~out.bool()).float() * ctx.slope * grad_input
        )
        return grad
    
def LSO(slope=0.1):
    """
    Wrapper of LeakySpikeOperator
    """
    slope = slope
    
    def inner(x):
        return StochasticSpikeOperator.apply(x, slope)
    
    return inner

class SparseFastSigmoid(torch.autograd.Function):
    """
    Computes the sparse and fast sigmoid function of the input tensor.
    """
    @staticmethod
    def forward(ctx, input_, slope=25, B=1):
        """
        Performs the forward pass of the SparseFastSigmoid function.
        
        Args:
        - input_ (torch.Tensor): Input tensor to be transformed.
        - slope (float): Slope of the sigmoid function.
        - B (float): Threshold of the sparse component.
        
        Returns:
        - torch.Tensor: Result of the SparseFastSigmoid function applied to the input tensor.
        """
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.B = B
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs the backward pass of the SparseFastSigmoid function.
        
        Args:
        - grad_output (torch.Tensor): Gradient of the output tensor.
        
        Returns:
        - Tuple of torch.Tensors: Gradients of the input tensor, slope, and B respectively.
        """
        (input_, ) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2 * (input_ > ctx.B).float()
        )
        return grad, None, None
    
def SFS(slope=25, B=1):
    """
    Wrapper of SparseFastSigmoid
    """
    slope = slope
    B = B
    
    def inner(x):
        return SparseFastSigmoid.apply(x, slope, B)
    
    return inner

class FractionalOrderGradient(torch.autograd.Function):
    """
    This class defines a PyTorch autograd function that applies a fractional 
    order gradient operation to its input tensor.
    """

    @staticmethod
    def forward(ctx, input_, alpha=0.9):
        """
        Computes the forward pass of the function.
        
        Args:
            ctx: a context object to save tensors and other variables needed 
                 for the backward pass
            input_: an input tensor
            alpha: the fractional order parameter (default=0.9)
            
        Returns:
            A tensor resulting from applying a thresholding operation to the 
            input tensor.
        """
        ctx.save_for_backward(input_)
        ctx.gamma_value = math.gamma(1 - alpha)
        ctx.alpha = alpha
        out = (input_ > 0).float()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradients of the function with respect to its inputs.
        
        Args:
            ctx: a context object containing tensors and other variables saved 
                 during the forward pass
            grad_output: a tensor containing the gradients of the output tensor 
                         with respect to some downstream variable
                         
        Returns:
            A tuple of gradient tensors for each input argument. Since this 
            function has only one input argument, it returns a tuple with one 
            element: the gradient of the function with respect to the input 
            tensor.
        """
        grad_input = grad_output.clone()
        (input_, ) = ctx.saved_tensors
        grad = (
            1 / (ctx.gamma_value * torch.pow(input_, ctx.alpha))
        )
        return grad, None, None
    
def FOG(alpha=0.9):
    """
    Wrapper of FractionalOrderGradient
    """
    alpha = alpha
    
    def inner(x):
        return FractionalOrderGradient.apply(x, alpha)
    
    return inner

