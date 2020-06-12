from abc import ABC, abstractmethod


class BaseLoss(ABC):

    def __init__(self, name="loss_function"):

        self.name = name

    def __call__(self, M_pred, **kwargs):
        return self._build_loss(M_pred, **kwargs)

    @abstractmethod
    def _build_loss(self, M_pred, **kwargs):
        pass

    