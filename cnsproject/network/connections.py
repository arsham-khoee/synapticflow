"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

from neural_populations import NeuralPopulation


class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `w_min` and `w_max`. \
    Also,  if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    w:  Tensor
        The tensor of weights for the connection.
    d: Tensor
       The tensor of delays for the connection.
    d_min: float
        The minimum possible delay. The default is 0.0.
    d_max: float
        The maximum possible delay. The default is 100.
    mask:  ByteTensor
        Define a mask to determine which weights to clamp to zero.

    Keyword Arguments
    -----------------
    
    w_min : float
        The minimum possible synaptic strength. The default is 0.0.
    w_max : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.
    b: Tensor
        bias parameter
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        w: torch.Tensor,
        d: torch.Tensor,
        d_min: float = 0.0,
        d_max: float = 100.0,
        mask: torch.ByteTensor = True,
        **kwargs
    ) -> None:
        super().__init__()

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post

        self.w = w

        self.w_min = kwargs.get('w_min', torch.zeros(pre.n, post.n))
        self.w_max = kwargs.get('w_max', torch.ones(pre.n, post.n))
        self.norm = kwargs.get('norm', None)
        
        self.d_min = d_min
        self.d_max = d_max
        """
        define memories to save sequence of weights and delays for spikes.  
        """
        self.delay_mem = torch.Tensor([])
        self.w_mem = torch.Tensor([])

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        pass
    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass
    
class Connection(AbstractConnection):
    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        w: torch.Tensor = None,
        d: torch.Tensor = None,
        d_min: float = 0.0,
        d_max: float = 100.0,
        mask: torch.ByteTensor = True,
        **kwargs
    ) -> None:
        super().__init__(
            pre = pre,
            post = post,
            w = w,
            d = d,
            d_min = d_min,
            d_max = d_max,
            mask = mask
        )

        if w is None:
            if (self.w_min == float('-inf')).any() or (self.w_max == float('inf')).any():
                w = torch.clamp(torch.rand(pre.n, post.n), self.w_min, self.w_max)
            else:
                w = self.w_min + torch.rand(pre.n, post.n) * (self.w_max - self.w_min)
        else:
            if (self.w_min != float('-inf')).any() or (self.w_max != float('inf')).any():
                w = torch.clamp(torch.as_tensor(w), self.w_min, self.w_max)

        if d is None:
            if (self.d_min == 0.0) or (self.d_max == float('inf')):
                d = torch.clamp(torch.rand(pre.n, post.n), self.d_min, self.d_max)
            else:
                d = self.d_min + torch.rand(pre.n, post.n) * (self.d_max - self.d_min)
        else:
            if (self.d_min != 0.0) or (self.d_max != float('inf')):
                d = torch.clamp(torch.as_tensor(d), self.d_min, self.d_max)     

        self.w = Parameter(w, requires_grad=False)
        self.d = Parameter(d, requires_grad=False)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """

        self.delay_mem = torch.cat((self.delay_mem, torch.masked_select(self.d, s.bool())), 0)
        self.w_mem = torch.cat((self.w_mem, torch.masked_select(self.w, s.bool())), 0)

        exp = self.delay_mem <= 0
        w_result = self.w_mem.dot(exp.float())

        self.w_mem = torch.masked_select(self.w_mem, torch.logical_not(exp))
        self.delay_mem = self.delay_mem > 0
        

        if self.b is None:
            result = torch.ones(s.size(0), *self.post.shape) * w_result
        else:
            result = torch.ones(s.size(0), *self.post.shape) * w_result + self.b 

        return result  
        #return torch.ones(s.size(0), *self.post.shape) * w_result

    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)
        

class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    Implement the dense connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class Conv1dConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        kernel_size: int,
        w: torch.Tensor = None,
        weight_decay: float = 0.0,
        stride: int = 1,
        padding: int = 0,
        dilation = 0, 
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            weight_decay=weight_decay,
            **kwargs
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels, input_size = (self.pre.shape[0], self.pre.shape[1])
        self.out_channels, output_size = (self.post.shape[0], self.post.shape[1])

        conv_size = (input_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        shape = (self.in_channels, self.out_channels, int(conv_size))

        error = (
            "Target dimensionality must be (out_channels, ?,"
            "(input_size - filter_size + 2 * padding) / stride + 1,"
        )

        assert self.post.shape[0] == shape[1] and self.post.shape[1] == shape[2], error

        if w is None:
            if (self.w_min == float('-inf')).any() or (self.w_max == float('inf')).any():
                w = torch.clamp(torch.rand(self.post.shape[0], self.pre.shape[0], self.kernel_size), self.w_min, self.w_max)
            else:
                w = self.w_min + torch.rand(self.post.shape[0], self.pre.shape[0], self.kernel_size) * (self.w_max - self.w_min)
                print('w:' , w)
        else:
            if (self.w_min != float('-inf')).any() or (self.w_max != float('inf')).any():
                w = torch.clamp(torch.as_tensor(w), self.w_min, self.w_max)

        self.w = Parameter(w, requires_grad=False)

        self.b = Parameter(
            kwargs.get("b", torch.zeros(self.out_channels)), requires_grad=False
        )

        """
        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        return F.conv1d(
            s.float(),
            self.w,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        """
        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        super().reset_state_variables()

class Conv2dConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        kernel_size: Union[int, Tuple[int, int]],
        w: torch.Tensor = None,
        weight_decay: float = 0.0,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            weight_decay=weight_decay,
            **kwargs
        )
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_channels, input_height, input_width = (self.pre.shape[0], self.pre.shape[1], self.pre.shape[2])
        self.out_channels, output_height, output_width = (self.post.shape[0], self.post.shape[1], self.post.shape[2])

        height = (input_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        width =  (input_width - self.kernel_size [1]+ 2 * self.padding[1]) / self.stride[1] + 1
        shape = (self.in_channels, self.out_channels, int(height), int(width))
        print(shape)
        print('height', height)
        print('width', width)
        error = (
            "Target dimensionality must be (out_channels, ?,"
            "(input_height - filter_height + 2 * padding_height) / stride_height + 1,"
            "(input_width - filter_width + 2 * padding_width) / stride_width + 1"
        )


        print(self.post.shape[0] == shape[1] , self.post.shape[1] == shape[2] , self.post.shape[2] == shape[3])
        assert self.post.shape[0] == shape[1] and self.post.shape[1] == shape[2] and self.post.shape[2] == shape[3], error
        if w is None:
            if (self.w_min == float('-inf')).any() or (self.w_max == float('inf')).any():
                w = torch.clamp(torch.rand(self.post.shape[0], self.pre.shape[0], *self.kernel_size), self.w_min, self.w_max)
            else:
                w = self.w_min + torch.rand(self.post.shape[0], self.pre.shape[0], *self.kernel_size) * (self.w_max - self.w_min)
                print('w:' , w)
        else:
            if (self.w_min != float('-inf')).any() or (self.w_max != float('inf')).any():
                w = torch.clamp(torch.as_tensor(w), self.w_min, self.w_max)

        self.w = Parameter(w, requires_grad=False)

        self.b = Parameter(
            kwargs.get("b", torch.zeros(self.out_channels)), requires_grad=False
        )

        """
        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        print('b', self.b)
        return F.conv2d(
            s.float(),
            self.w,
            self.b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        """
        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        super().reset_state_variables()    

class PoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection\
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to\
    make some modifications in the defined structure of this class.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of post-synaptic population activity given the
        activity of the pre-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.\
        You might need to call the parent method.

        Note: You should be careful with this method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass
