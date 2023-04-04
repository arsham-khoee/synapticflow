"""
Module for reward dynamics.

TODO.

Define your reward functions here.
"""

from abc import ABC, abstractmethod


class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    Make sure to implement the abstract methods in your child class.

    To implement your dopamine functionality, You will write a class \
    inheriting this abstract class. You can add attributes to your \
    child class. The dynamics of dopamine function (DA) will be \
    implemented in `compute` method. So you will call `compute` in \
    your reward-modulated learning rules to retrieve the dopamine \
    value in the desired time step. To reset or update the defined \
    attributes in your reward function, use `update` method and \
    remember to call it your learning rule computations in the \
    right place.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass
