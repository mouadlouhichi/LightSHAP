"""Pipeline package.

Import stages without pulling the heavy runner, so
`from lightshap.pipeline.stages import STAGE_ORDER` stays cheap.
"""

from lightshap.pipeline.stages import STAGE_ORDER

__all__ = ["STAGE_ORDER", "PipelineRunner", "run_from_config"]


def __getattr__(name: str):
    if name in {"PipelineRunner", "run_from_config"}:
        from lightshap.pipeline.runner import PipelineRunner, run_from_config

        return PipelineRunner if name == "PipelineRunner" else run_from_config
    raise AttributeError(name)
