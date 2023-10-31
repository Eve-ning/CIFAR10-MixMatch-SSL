from typing import Callable


def epoch_scaler(
    scale_from: float, scale_to: float
) -> Callable[[int, int], float]:
    def fn(epoch: int, n_epochs: int) -> float:
        epoch_frac = epoch / n_epochs
        return scale_from + (scale_to - scale_from) * epoch_frac

    return fn


def ema_decayer(
    decay_from: float, decay_to: float
) -> Callable[[int, int], float]:
    def fn(epoch: int, n_epochs: int) -> float:
        epoch_frac = epoch / n_epochs
        return decay_from + (decay_to - decay_from) * epoch_frac

    return fn
