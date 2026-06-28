"""MASW: Multi-Agent Shared Write defense prototype."""

from .pipeline import SecureSharedWritePipeline, build_default_pipeline
from .types import Agent, Task, TaskContext, TrustLevel

__all__ = [
    "Agent",
    "SecureSharedWritePipeline",
    "Task",
    "TaskContext",
    "TrustLevel",
    "build_default_pipeline",
]
