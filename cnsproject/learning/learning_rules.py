"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence

import numpy as np
import torch

from ..network.connections import AbstractConnection


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
        **kwargs
    ) -> None:
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.connection = connection

        self.lr = torch.tensor(lr, dtype=torch.float)

        self.weight_decay = 1 - weight_decay if weight_decay else 1.

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay
        # if (
        #     self.connection.w_min != -np.inf or self.connection.w_max != np.inf
        # ) and not isinstance(self.connection, NoOp):
        #     self.connection.w.clamp_(self.connection.w_min,
        #                              self.connection.w_max)


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
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
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
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        
        dw = self.connection.pre.dt * (-self.lr[0] * self.connection.post.traces.view(*self.connection.post.shape, 1).matmul(self.connection.pre.s.view(1, *self.connection.pre.shape).float()).T + (self.lr[1] * self.connection.pre.traces.view(*self.connection.pre.shape, 1).matmul(self.connection.post.s.view(1, *self.connection.post.shape).float())))
        
        self.connection.w += dw * (1 - self.connection.w)

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
        weight_decay: float = 0.,
        window_steps: int = 10,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
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
        weight_decay: float = 0.,
        tau_c: Union[float, torch.Tensor] = 0.1,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
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
        self.connection.w += dw.T
        """

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        super().update()


class FlatRSTDP(LearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass
