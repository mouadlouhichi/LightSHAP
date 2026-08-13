"""Typed failures. Never swallow scientific violations."""


class LightShapError(Exception):
    """Base error."""


class ConfigError(LightShapError):
    """Invalid or unfrozen configuration."""


class FrozenInvariantError(ConfigError):
    """A scientific freeze was violated."""


class ArtifactError(LightShapError):
    """Missing, corrupt, or schema-invalid artifact."""


class CheckpointError(LightShapError):
    """Checkpoint read/write/compatibility failure."""


class StageError(LightShapError):
    """Pipeline stage refused to run or failed validation."""


class DependencyError(StageError):
    """A required upstream stage is missing or invalid."""


class DataIntegrityError(LightShapError):
    """Dataset hash / schema / leakage check failed."""


class MathInvariantError(LightShapError):
    """A mathematical identity (efficiency, fixture, …) failed."""
