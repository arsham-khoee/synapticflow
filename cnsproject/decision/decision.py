"""
Module for decision making.

TODO.

1. Implement the dynamics of decision making. You are free to define your
   structure here.
2. Make sure to implement winner-take-all mechanism.
"""

from abc import ABC, abstractmethod


class AbstractDecision(ABC):
    """
    Abstract class to define decision making strategy.

    Make sure to implement the abstract methods in your child class.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the variables after making the decision.

        Returns
        -------
        None

        """
        pass


class WinnerTakeAllDecision(AbstractDecision):
    """
    The k-Winner-Take-All decision mechanism.

    You will have to define a constructor and specify the required \
    attributes, including k, the number of winners.
    """

    def compute(self, **kwargs) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        pass

    def update(self, **kwargs) -> None:
        """
        Update the variables after making the decision.

        Returns
        -------
        None

        """
        pass
