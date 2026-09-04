import os
import importlib
import pkgutil
from typing import Dict, Type, List
from .exceptions import PluginNotFoundError, DuplicatePluginError, PluginNotFoundException


class PluginRegistry:
    """
    Central Plugin Registry with dynamic discovery.
    Supports decorators: @register_pruner, @register_criterion,
    @register_granularity, @register_model_adapter, @register_selector.
    """

    _PRUNERS: Dict[str, Type] = {}
    _CRITERIA: Dict[str, Type] = {}
    _GRANULARITIES: Dict[str, Type] = {}
    _MODELS: Dict[str, Type] = {}
    _SELECTORS: Dict[str, Type] = {}

    @classmethod
    def register_pruner(cls, name: str, override: bool = False):
        def decorator(subclass):
            key = name.lower()
            if not override and key in cls._PRUNERS and cls._PRUNERS[key] != subclass:
                raise DuplicatePluginError(f"Pruner '{name}' is already registered by {cls._PRUNERS[key]}. Cannot register {subclass}.")
            cls._PRUNERS[key] = subclass
            return subclass
        return decorator

    @classmethod
    def register_criterion(cls, name: str, override: bool = False):
        def decorator(subclass):
            key = name.lower()
            if not override and key in cls._CRITERIA and cls._CRITERIA[key] != subclass:
                raise DuplicatePluginError(f"Criterion '{name}' is already registered by {cls._CRITERIA[key]}. Cannot register {subclass}.")
            cls._CRITERIA[key] = subclass
            return subclass
        return decorator

    @classmethod
    def register_granularity(cls, name: str, override: bool = False):
        def decorator(subclass):
            key = name.lower()
            if not override and key in cls._GRANULARITIES and cls._GRANULARITIES[key] != subclass:
                raise DuplicatePluginError(f"Granularity '{name}' is already registered by {cls._GRANULARITIES[key]}. Cannot register {subclass}.")
            cls._GRANULARITIES[key] = subclass
            return subclass
        return decorator

    @classmethod
    def register_model_adapter(cls, name: str, override: bool = False):
        def decorator(subclass):
            key = name.lower()
            if not override and key in cls._MODELS and cls._MODELS[key] != subclass:
                raise DuplicatePluginError(f"Model Adapter '{name}' is already registered by {cls._MODELS[key]}. Cannot register {subclass}.")
            cls._MODELS[key] = subclass
            return subclass
        return decorator

    @classmethod
    def register_selector(cls, name: str, override: bool = False):
        def decorator(subclass):
            key = name.lower()
            if not override and key in cls._SELECTORS and cls._SELECTORS[key] != subclass:
                raise DuplicatePluginError(f"Selector '{name}' is already registered by {cls._SELECTORS[key]}. Cannot register {subclass}.")
            cls._SELECTORS[key] = subclass
            return subclass
        return decorator

    @classmethod
    def get_pruner(cls, name: str) -> Type:
        cls._ensure_plugins_loaded()
        key = name.lower()
        if key not in cls._PRUNERS:
            raise PluginNotFoundException(f"Pruner '{name}' not found. Registered: {list(cls._PRUNERS.keys())}")
        return cls._PRUNERS[key]

    @classmethod
    def get_criterion(cls, name: str) -> Type:
        cls._ensure_plugins_loaded()
        key = name.lower()
        if key not in cls._CRITERIA:
            raise PluginNotFoundException(f"Criterion '{name}' not found. Registered: {list(cls._CRITERIA.keys())}")
        return cls._CRITERIA[key]

    @classmethod
    def get_granularity(cls, name: str) -> Type:
        cls._ensure_plugins_loaded()
        key = name.lower()
        if key not in cls._GRANULARITIES:
            raise PluginNotFoundException(f"Granularity '{name}' not found. Registered: {list(cls._GRANULARITIES.keys())}")
        return cls._GRANULARITIES[key]

    @classmethod
    def get_model_adapter(cls, name: str) -> Type:
        cls._ensure_plugins_loaded()
        key = name.lower()
        if key not in cls._MODELS:
            raise PluginNotFoundException(f"Model Adapter '{name}' not found. Registered: {list(cls._MODELS.keys())}")
        return cls._MODELS[key]

    @classmethod
    def get_selector(cls, name: str) -> Type:
        cls._ensure_plugins_loaded()
        key = name.lower()
        if key not in cls._SELECTORS:
            raise PluginNotFoundException(f"Selector '{name}' not found. Registered: {list(cls._SELECTORS.keys())}")
        return cls._SELECTORS[key]

    @classmethod
    def list_pruners(cls) -> List[str]:
        cls._ensure_plugins_loaded()
        return list(cls._PRUNERS.keys())

    @classmethod
    def list_criteria(cls) -> List[str]:
        cls._ensure_plugins_loaded()
        return list(cls._CRITERIA.keys())

    @classmethod
    def list_granularities(cls) -> List[str]:
        cls._ensure_plugins_loaded()
        return list(cls._GRANULARITIES.keys())

    @classmethod
    def list_models(cls) -> List[str]:
        cls._ensure_plugins_loaded()
        return list(cls._MODELS.keys())

    @classmethod
    def list_selectors(cls) -> List[str]:
        cls._ensure_plugins_loaded()
        return list(cls._SELECTORS.keys())

    @classmethod
    def _ensure_plugins_loaded(cls):
        """Dynamically imports all modules under prune_framework.plugins to trigger decorator registration."""
        import prune_framework.plugins as plugins_pkg
        plugins_dir = os.path.dirname(plugins_pkg.__file__)
        for sub_dir in ["pruners", "criteria", "granularities", "adapters", "selectors"]:
            sub_path = os.path.join(plugins_dir, sub_dir)
            if os.path.isdir(sub_path):
                for file in os.listdir(sub_path):
                    if file.endswith(".py") and not file.startswith("__"):
                        module_name = f"prune_framework.plugins.{sub_dir}.{file[:-3]}"
                        importlib.import_module(module_name)


# Shorthand alias decorators
register_pruner = PluginRegistry.register_pruner
register_criterion = PluginRegistry.register_criterion
register_granularity = PluginRegistry.register_granularity
register_model = PluginRegistry.register_model_adapter
register_selector = PluginRegistry.register_selector
