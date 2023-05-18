"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence

import numpy as np
import torch
import copy

from ..network.connections import AbstractConnection, SparseConnection, RandomConnection


class LearningRule(ABC):
    """
    Abstract class for defining learning rules.

    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.

    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        reduction: Optional[callable] = None,
        boundry: str = 'hard',
        **kwargs
    ) -> None:
        
        assert not isinstance(connection, RandomConnection), 'RandomConnection is not learnable!'
        
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.connection = connection
        self.boundry = boundry

        self.lr = torch.tensor(lr, dtype=torch.float)

        self.weight_decay = 1 - weight_decay if weight_decay else 1.
        
        if reduction is None:
            if self.connection.pre.batch_size == 1:
                self.reduction = torch.squeeze
            else:
                self.reduction = torch.sum
        else:
            self.reduction = reduction

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay
        if (
            self.connection.w_min != -np.inf or self.connection.w_max != np.inf
        ) and not isinstance(self, NoOp) and (self.boundry == 'hard'):
            self.connection.w.clamp_(self.connection.w_min,
                                     self.connection.w_max)
        
        if isinstance(self.connection, SparseConnection):
            self.connection.w *= self.connection.sparse_mask

class NoOp(LearningRule):
    """
    Learning rule with no effect.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.

        Returns
        -------
        None

        """
        super().update()

class STDP(LearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundry: str = 'hard',
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            reduction=reduction,
            weight_decay=weight_decay,
            boundry=boundry,
            **kwargs
        )
        """
        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        
        dw = self.connection.pre.dt * (-self.lr[0] * self.connection.post.traces.view(*self.connection.post.shape, 1).matmul(self.connection.pre.s.view(1, *self.connection.pre.shape).float()).T + (self.lr[1] * self.connection.pre.traces.view(*self.connection.pre.shape, 1).matmul(self.connection.post.s.view(1, *self.connection.post.shape).float())))
        
        if self.boundry == 'soft':
            self.connection.w += (dw * ((self.connection.w - self.connection.w_min) * (self.connection.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min)))
        else:
            self.connection.w += dw

        super().update()

        """
        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """

class FlatSTDP(LearningRule):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundry: str = 'hard',
        window_steps: int = 10,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            reduction=reduction,
            weight_decay=weight_decay,
            boundry=boundry,
            **kwargs
        )
        self.window_steps = window_steps
        self.pre_traces = torch.zeros(window_steps, *connection.pre.shape)
        self.post_traces = torch.zeros(window_steps, *connection.post.shape)
        self.pre_traces_i = 0
        self.post_traces_i = 0
        """

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        
        self.pre_traces_i %= self.window_steps
        self.pre_traces[self.pre_traces_i] = self.connection.pre.s
        self.pre_traces_i += 1
        pre_traces_sum = sum(self.pre_traces)
        
        
        print(self.pre_traces_i)
        print(self.pre_traces)
        print(pre_traces_sum)
        
        self.post_traces_i %= self.window_steps
        self.post_traces[self.post_traces_i] = self.connection.post.s
        self.post_traces_i += 1
        post_traces_sum = sum(self.post_traces)
        
        
        print(self.post_traces_i)
        print(self.post_traces)
        print(post_traces_sum)
       
        dw = self.connection.pre.dt * (-self.lr[0] * post_traces_sum.view(*self.connection.post.shape, 1).matmul(self.connection.pre.s.float().view(1, *self.connection.pre.shape)).T + self.lr[1] * pre_traces_sum.view(*self.connection.pre.shape, 1).matmul(self.connection.post.s.float().view(1, *self.connection.post.shape)))
        
        if self.boundry == 'soft':
            self.connection.w += (dw * ((self.connection.w - self.connection.w_min) * (self.connection.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min)))
        else:
            self.connection.w += dw

        """

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        super().update()

class RSTDP(LearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundry: str = 'hard',
        tau_c: Union[float, torch.Tensor] = 0.1,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            reduction=reduction,
            weight_decay=weight_decay,
            boundry=boundry,
            **kwargs
        )
        self.c = torch.zeros(*connection.post.shape,*connection.pre.shape)
        #self.register_buffer("tau_c", torch.tensor(tau_c, dtype=torch.float))

        self.tau_c = tau_c
        """
        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, da:float, **kwargs) -> None:
        
        delta = self.connection.post.s.float().view(*self.connection.post.shape, 1).matmul(self.connection.pre.s.float().view(1, *self.connection.pre.shape))
        
        dc = (-self.c / self.tau_c) * self.connection.pre.dt + (self.connection.pre.dt * (-self.lr[0] * self.connection.post.traces.view(*self.connection.post.shape, 1).matmul(self.connection.pre.s.view(1, *self.connection.pre.shape).float()) + (self.lr[1] * self.connection.pre.traces.view(*self.connection.pre.shape, 1).matmul(self.connection.post.s.view(1, *self.connection.post.shape).float())).T) * delta)
        self.c += dc
        
        
        dw = self.connection.pre.dt * (self.c * da)
        dw = dw.T
        
        if self.boundry == 'soft':
            self.connection.w += (dw * ((self.connection.w - self.connection.w_min) * (self.connection.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min)))
        else:
            self.connection.w += dw
            
        """

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        super().update()

class DSTDP(LearningRule):
    """
    Delay Related STDP implementation from Bio-plausible Unsupervised Delay Learning for Extracting Temporal Features in Spiking Neural Networks paper.
    Paper link: https://arxiv.org/pdf/2011.09380.pdf
    """
    
    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundry: str = 'hard',
        delay_learning: bool = False,
        A_positive: float = 1,
        A_negative: float = 1,
        tau_positive: float = 1,
        tau_negative: float = 1,
        B_positive: float = 1,
        B_negative: float = 1,
        sigma_positive: float = 1,
        sigma_negative: float = 1,
        **kwargs
    ) -> None:
        
        """
        DSDTP Constructor
        
        Params:
        -------
        connection (Connection) : A connection to apply learning rule.
        lr (Optional(float / Sequence(float))) : Learning rate for both pre and post synaptic neurons. Default: None
        weight_decay (float) : Weight decay for learning process. Default: 0
        delay_learning (bool) : Indicates apply delay learning for delay parameter or not. Default: False
        A_positive (float) : Constant for weights changes function. Default: 1
        A_negative (float) : Constant for weights changes function. Default: 1
        tau_positive (float) : Constant for weights changes exponential rate. Default: 1
        tau_negative (float) : Constant for weights changes exponential rate. Default: 1
        B_positive (float) : Constant for delay changes function. Default: 1
        B_negative (float) : Constant for delay changes function. Default: 1
        sigma_positive (float) : Constant for delay changes exponential rate. Default: 1
        sigma_negative (float) : Constant for delay changes exponential rate. Default: 1
        """
        
        super().__init__(
            connection=connection,
            lr=lr,
            reduction=reduction,
            weight_decay=weight_decay,
            boundry=boundry
            **kwargs
        )
        
        self.delay_learning = delay_learning
        self.delay_mem = torch.zeros((self.connection.pre.n, self.connection.post.n), dtype=torch.float32)
        self.A_positive = A_positive
        self.A_negative = A_negative
        self.tau_positive = tau_positive
        self.tau_negative = tau_negative
        self.B_positive = B_positive
        self.B_negative = B_negative
        self.sigma_positive = sigma_positive
        self.sigma_negative = sigma_negative
        
    def update(self, **kwargs):
        
        # Simulate one time step and compute required information
        self.delay_mem[self.delay_mem.nonzero(as_tuple=True)] += self.connection.pre.dt
        pos_s = self.connection.post.s
        pos_s = pos_s.T.repeat(self.connection.pre.n, 1)
        delta_time = pos_s.mul(self.delay_mem)
        delta_w = torch.zeros_like(self.connection.w)
        
        # Checks if any modification should be applied on weights
        delta_w = self.F(delta_time) * (float(delta_time.any()))
        
        # Checks if any modification should be applied on delays
        delta_d = self.G(delta_time) * (float(self.delay_learning))
        
        # Apply modification on weights and delays
        if self.boundry == 'soft':
            self.connection.w += delta_w * ((self.connection.w - self.connection.w_min) * (self.connection.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min))
        else:
            self.connection.w += delta_w
            
        if self.boundry == 'soft':
            self.connection.d += delta_d * ((self.connection.d - self.connection.d_min) * (self.connection.d_max - self.connection.d) / (self.connection.d_max - self.connection.d_min))
        else:
            self.connection.d += delta_d
        
        # Remove old memory
        self.delay_mem.masked_fill_(delta_time != 0, 0)
        
        # Indicates new potential modification
        pre_s = self.connection.pre.s
        pre_s = pre_s.repeat(self.connection.post.n, 1).T
        result = pre_s.mul(self.connection.d)
        self.delay_mem.masked_fill_(result != 0, 0)
        result *= -1
        self.delay_mem += result
        
        super().update()
        
    def F(self, delta_time : torch.Tensor) -> torch.Tensor:
        """
        Non-Linear exponential function for indicating change in weight
        
        Params:
        -------
        delta_time (torch.tensor) : delta time = |t_j - t_i - d_ij|
        
        Returns:
        -------
        torch.Tensor : Output of this function
        """
        result = copy.deepcopy(delta_time)
        result[result.nonzero(as_tuple=True)] = (result[result.nonzero(as_tuple=True)] >= 0).float() * (self.A_positive * torch.exp((-1 * result[result.nonzero(as_tuple=True)]) / self.tau_positive)) + (result[result.nonzero(as_tuple=True)] < 0 ).float() * (-1 * self.A_negative * torch.exp((result[result.nonzero(as_tuple=True)]) / self.tau_negative))
        return result

    def G(self, delta_time : torch.Tensor) -> torch.Tensor:
        """
        Non-Linear exponential function for indicating change in weight
        
        Params:
        -------
        delta_time (torch.tensor) : delta time = |t_j - t_i - d_ij|
        
        Returns:
        -------
        torch.Tensor : Output of this function
        """
        result = copy.deepcopy(delta_time)
        result[result.nonzero(as_tuple=True)] = (result[result.nonzero(as_tuple=True)] >= 0).float() * (-1 * self.B_negative * torch.exp((-1 * result[result.nonzero(as_tuple=True)]) / self.sigma_negative)) + (result[result.nonzero(as_tuple=True)] < 0).float() * (self.B_positive * torch.exp((result[result.nonzero(as_tuple=True)]) / self.sigma_positive))
        return result
    
class MNSTDP(LearningRule):
    """
    Mobin Nesari STDP implementation from Bio-plausible Unsupervised Delay Learning for Extracting Temporal Features in Spiking Neural Networks paper.
    """
    
    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundry: str = 'hard',
        A_positive: float = 0.2,
        A_negative: float = 0.2,
        tau_positive: float = 0.01,
        tau_negative: float = 0.01,
        **kwargs
    ) -> None:
        
        """
        DSDTP Constructor
        
        Params:
        -------
        connection (Connection) : A connection to apply learning rule.
        lr (Optional(float / Sequence(float))) : Learning rate for both pre and post synaptic neurons. Default: None
        weight_decay (float) : Weight decay for learning process. Default: 0
        A_positive (float) : Constant for weights changes function. Default: 1
        A_negative (float) : Constant for weights changes function. Default: 1
        tau_positive (float) : Constant for weights changes exponential rate. Default: 1
        tau_negative (float) : Constant for weights changes exponential rate. Default: 1
        """
        
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            boundry=boundry,
            reduction=reduction,
            **kwargs
        )
        
        self.weight_mem = torch.zeros((self.connection.pre.n, self.connection.post.n), dtype=torch.float32)
        self.A_positive = A_positive
        self.A_negative = A_negative
        self.tau_positive = tau_positive
        self.tau_negative = tau_negative
        
    def update(self, **kwargs):
        
        # Simulate one time step and compute required information
        self.weight_mem[self.weight_mem.nonzero(as_tuple=True)] += self.connection.pre.dt
        pos_s = self.connection.post.s
        pos_s = pos_s.T.repeat(self.connection.pre.n, 1)
        delta_time = pos_s.mul(self.weight_mem)
        delta_time[delta_time.nonzero(as_tuple=True)] += 1
        delta_w = torch.zeros_like(self.connection.w)
        
        # Checks if any modification should be applied on weights
        delta_w = self.F(delta_time) * (float(delta_time.any()))
        
        # Apply modification on weights and delays
        if self.boundry == 'soft':
            self.connection.w += delta_w * ((self.connection.w - self.connection.w_min) * (self.connection.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min))
        else:
            self.connection.w += delta_w
        
        # Remove old memory
        self.weight_mem.masked_fill_(delta_time != 0, 0)
        
        # Indicates new potential modification
        pre_s = self.connection.pre.s
        pre_s = pre_s.repeat(self.connection.post.n, 1).T
        result = pre_s.mul(torch.ones_like(self.connection.d))
        self.weight_mem.masked_fill_(result != 0, 0)
        result *= -1
        self.weight_mem += result
        
        super().update()
        
    def F(self, delta_time : torch.Tensor) -> torch.Tensor:
        """
        Non-Linear exponential function for indicating change in weight
        
        Params:
        -------
        delta_time (torch.tensor) : delta time = |t_j - t_i - d_ij|
        
        Returns:
        -------
        torch.Tensor : Output of this function
        """
        result = copy.deepcopy(delta_time)
        result[result.nonzero(as_tuple=True)] = (result[result.nonzero(as_tuple=True)] >= 0).float() * (self.A_positive * torch.exp((-1 * result[result.nonzero(as_tuple=True)]) / self.tau_positive)) + (result[result.nonzero(as_tuple=True)] < 0 ).float() * (-1 * self.A_negative * torch.exp((result[result.nonzero(as_tuple=True)]) / self.tau_negative))
        return result
    
class WeightDependent(LearningRule):
    """
    Weight Dependent STDP Learning Rule
    """
    
    def __init__(self,
                 connection: AbstractConnection, 
                 lr: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
                 reduction: Optional[callable] = None,
                 weight_decay: float = 0.0,
                 boundry: str = 'hard',
                 **kwargs
                 ) -> None:
        super().__init__(
                         connection=connection,
                         lr=lr,
                         reduction=reduction,
                         weight_decay=weight_decay,
                         boundry=boundry
                         **kwargs)
        
        assert self.connection.pre.spike_trace, "Pre-synaptic population should record spike traces."
        assert (connection.w_min != -np.inf and connection.w_max != np.inf), "Connection should define finite w_min and w_max"
        
        self.w_min = connection.w_min
        self.w_max = connection.w_max
        
        
    def update(self, **kwargs) -> None:
        batch_size = self.connection.pre.batch_size
        
        pre_s = self.connection.pre.s.view(batch_size, -1).unsqueeze(2).float()
        pre_trace = self.connection.pre.traces.view(batch_size, -1).unsqueeze(2)
        post_s = self.connection.post.s.view(batch_size, -1).unsqueeze(1).float()
        post_trace = self.connection.post.traces.view(batch_size, -1).unsqueeze(1)
        
        update = 0
        
        if self.lr[0].any():
            outer_product = self.reduction(torch.bmm(pre_s, post_trace), dim=0)
            update -= self.lr[0] * outer_product * (self.connection.w - self.w_min)
            
        if self.lr[1].any():
            outer_product = self.reduction(torch.bmm(pre_trace, post_s), dim=0)
            update += self.lr[1] * outer_product * (self.w_max - self.connection.w)
            
        if self.boundry == 'soft':
            self.connection.w += update * ((self.connection.w - self.w_min) * (self.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min))
        else:
            self.connection.w += update
        
        super().update()
