from .base import Model
from .structural import Uniform, NashMix, LogitQRE, QuantalLevelK, CognitiveHierarchy


def baseline_ladder(include_neural: bool = True) -> list[Model]:
    """The models every new model must beat, from simplest to strongest."""
    models: list[Model] = [Uniform(), NashMix(), LogitQRE(), QuantalLevelK(), CognitiveHierarchy()]
    if include_neural:
        from .neural import MLP, ElementaryQLk, HybridQLk
        models += [ElementaryQLk(), MLP(), HybridQLk(residual=False), HybridQLk(residual=True)]
    return models
