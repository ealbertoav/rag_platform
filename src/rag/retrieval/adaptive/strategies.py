from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from opentelemetry import trace

from src.rag.retrieval.adaptive.query_classifier import QueryCategory

_DEFAULT_PARAMS: dict[QueryCategory, dict[str, int | bool]] = {
    QueryCategory.FACTUAL: {"top_k": 30, "n_variants": 1, "hyde": False, "compression": True},
    QueryCategory.ANALYTICAL: {"top_k": 50, "n_variants": 3, "hyde": True, "compression": True},
    QueryCategory.OPINION: {"top_k": 20, "n_variants": 2, "hyde": False, "compression": False},
    QueryCategory.CONTEXTUAL: {"top_k": 40, "n_variants": 2, "hyde": False, "compression": True},
}


@dataclass(frozen=True, slots=True)
class RetrievalStrategyParams:
    """Per-query retrieval tuning derived from the query category."""

    top_k: int
    n_variants: int
    hyde: bool
    compression: bool


class BaseRetrievalStrategy:
    """Category-specific retrieval parameters."""

    def __init__(self, category: QueryCategory, params: RetrievalStrategyParams) -> None:
        self.category = category
        self.params = params


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, params: RetrievalStrategyParams | None = None) -> None:
        super().__init__(
            QueryCategory.FACTUAL,
            params or _params_for_category(QueryCategory.FACTUAL),
        )


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, params: RetrievalStrategyParams | None = None) -> None:
        super().__init__(
            QueryCategory.ANALYTICAL,
            params or _params_for_category(QueryCategory.ANALYTICAL),
        )


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, params: RetrievalStrategyParams | None = None) -> None:
        super().__init__(
            QueryCategory.OPINION,
            params or _params_for_category(QueryCategory.OPINION),
        )


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    def __init__(self, params: RetrievalStrategyParams | None = None) -> None:
        super().__init__(
            QueryCategory.CONTEXTUAL,
            params or _params_for_category(QueryCategory.CONTEXTUAL),
        )


_StrategyFactory = Callable[[RetrievalStrategyParams], BaseRetrievalStrategy]

_STRATEGY_CLASSES: dict[QueryCategory, _StrategyFactory] = {
    QueryCategory.FACTUAL: FactualRetrievalStrategy,
    QueryCategory.ANALYTICAL: AnalyticalRetrievalStrategy,
    QueryCategory.OPINION: OpinionRetrievalStrategy,
    QueryCategory.CONTEXTUAL: ContextualRetrievalStrategy,
}


def _params_for_category(category: QueryCategory) -> RetrievalStrategyParams:
    raw = _DEFAULT_PARAMS[category]
    return RetrievalStrategyParams(
        top_k=int(raw["top_k"]),
        n_variants=int(raw["n_variants"]),
        hyde=bool(raw["hyde"]),
        compression=bool(raw["compression"]),
    )


def _params_from_config(raw: object) -> RetrievalStrategyParams:
    from src.core.settings import CategoryStrategySettings

    if isinstance(raw, CategoryStrategySettings):
        cfg = raw
    else:
        cfg = CategoryStrategySettings.model_validate(raw)
    return RetrievalStrategyParams(
        top_k=cfg.top_k,
        n_variants=cfg.n_variants,
        hyde=cfg.hyde,
        compression=cfg.compression,
    )


class AdaptiveStrategyRegistry:
    """Maps query categories to retrieval strategy parameters."""

    def __init__(self, strategies: dict[str, RetrievalStrategyParams] | None = None) -> None:
        configured = strategies or {}
        self._strategies: dict[str, RetrievalStrategyParams] = {}
        for category in QueryCategory:
            if category.value in configured:
                self._strategies[category.value] = configured[category.value]
            else:
                self._strategies[category.value] = _params_for_category(category)
        self._fallback = self._strategies[QueryCategory.FACTUAL.value]

    def get_strategy(self, category: str | None) -> BaseRetrievalStrategy:
        """Return a strategy for *category*; unknown values use the factual fallback."""
        key = (category or QueryCategory.FACTUAL.value).lower()
        params = self._strategies.get(key, self._fallback)
        try:
            query_category = QueryCategory(key)
        except ValueError:
            query_category = QueryCategory.FACTUAL
        strategy_cls = _STRATEGY_CLASSES.get(query_category, FactualRetrievalStrategy)
        return strategy_cls(params)

    def resolve_params(self, category: str | None) -> RetrievalStrategyParams:
        return self.get_strategy(category).params

    @classmethod
    def from_settings(cls) -> AdaptiveStrategyRegistry:
        from src.core.settings import settings

        cfg = settings.retrieval.adaptive
        params = {name: _params_from_config(raw) for name, raw in cfg.strategies.items()}
        return cls(strategies=params)


def record_strategy_span(category: str | None, params: RetrievalStrategyParams) -> None:
    """Attach adaptive strategy attributes to the current OTel span, if any."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.set_attribute("query.category", category or QueryCategory.FACTUAL.value)
    span.set_attribute("retrieval.strategy.top_k", params.top_k)
    span.set_attribute("retrieval.strategy.n_variants", params.n_variants)
    span.set_attribute("retrieval.strategy.hyde", params.hyde)
    span.set_attribute("retrieval.strategy.compression", params.compression)
