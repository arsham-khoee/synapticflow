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

        if (
            self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.w.clamp_(self.connection.wmin,
                                     self.connection.wmax)


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
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        pass


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

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        pass


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

class TRIPLET_STDP(LearningRule):
        
    def __init__(self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.,
        boundary: str = 'hard',
        **kwargs) -> None:
        super().__init__(connection=connection,
            lr=lr,
            reduction=reduction,
            weight_decay=weight_decay,
            boundary=boundary,
            **kwargs)
        
        self.A2_plus = torch.tensor(kwargs.get("A2_plus", 7.5e-10))
        self.A3_plus = torch.tensor(kwargs.get("A3_plus", 9.3e-3))
        self.A2_minus = torch.tensor(kwargs.get("A2_minus", 7e-3))
        self.A3_minus = torch.tensor(kwargs.get("A3_minus", 2.3e-4))

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 16.8))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 33.7))
        self.tc_x = torch.tensor(kwargs.get("tc_x", 101))
        self.tc_y = torch.tensor(kwargs.get("tc_y", 125))
        
    def update(self, **kwargs) -> None:
        batch_size = self.connection.pre.batch_size
        
        if not hasattr(self, "o_1"):
            self.o_1 = torch.zeros(
                batch_size,
                self.connection.post.n,
                device=self.connection.pre.s.device,
            )
        
        if not hasattr(self, "o_2"):
            self.o_2 = torch.zeros(
                batch_size,
                self.connection.post.n,
                device=self.connection.post.s.device
            )

        if not hasattr(self, "r_1"):
            self.r_1 = torch.zeros(
                batch_size,
                self.connection.pre.n,
                device=self.connection.post.s.device
            )
        
        if not hasattr(self, "r_2"):
            self.r_2 = torch.zeros(
                batch_size,
                self.connection.pre.n,
                device=self.connection.post.s.device
            )
            
        pre_s = self.connection.pre.s.view(batch_size, -1).float()
        post_s = self.connection.post.s.view(batch_size, -1).float()
        print(pre_s)
        print(post_s)
        update = -1 * self.lr[1] * self.o_1 * (self.A2_minus + self.A3_minus * self.r_2) + self.lr[0] * self.r_1 * (self.A2_plus + self.A3_plus * self.o_2)
        print(f"decrease: {- 1* self.lr[1] * self.o_1 * (self.A2_minus + self.A3_minus * self.r_2)}")
        print(f"increase: {self.lr[0] * self.r_1 * (self.A2_plus + self.A3_plus * self.o_2)}")
        dw =  self.reduction(update, dim = 0)
        
        self.r_1 *= torch.exp(-self.connection.pre.dt / self.tc_plus)
        self.r_1 += pre_s

        self.r_2 *= torch.exp(-self.connection.pre.dt / self.tc_x)
        self.r_2 += pre_s

        self.o_1 *= torch.exp(-self.connection.post.dt / self.tc_minus)
        self.o_1 += post_s

        self.o_2 *= torch.exp(-self.connection.post.dt / self.tc_y)
        self.o_2 += post_s

        if self.boundary == 'soft':
            self.connection.w += (dw * ((self.connection.w - self.connection.w_min) * (self.connection.w_max - self.connection.w) / (self.connection.w_max - self.connection.w_min)))
        else:
            self.connection.w += dw
        return dw

