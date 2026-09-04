class PruningFrameworkException(Exception):
    """Base exception for Pruning Framework."""
    pass


class DuplicatePluginError(PruningFrameworkException):
    """Raised when attempting to register a plugin with a name that is already registered."""
    pass


class PluginNotFoundError(PruningFrameworkException):
    """Raised when a requested plugin is not registered."""
    pass


class InvalidGranularityError(PruningFrameworkException):
    """Raised when an invalid granularity is requested for a pruner."""
    pass


class ModelAdapterError(PruningFrameworkException):
    """Raised when a model adapter fails to inspect or manipulate a model."""
    pass


class ConfigValidationError(PruningFrameworkException):
    """Raised when a framework YAML/dictionary config is invalid."""
    pass


class PruningExecutionError(PruningFrameworkException):
    """Raised when an error occurs during pruning execution."""
    pass


# Backward compatibility aliases
PluginNotFoundException = PluginNotFoundError
InvalidGranularityException = InvalidGranularityError
ModelAdapterException = ModelAdapterError
ConfigValidationException = ConfigValidationError

