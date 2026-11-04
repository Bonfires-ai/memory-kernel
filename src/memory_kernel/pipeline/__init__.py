"""The C2 build as a library: ``run_build(BuildConfig(...))``.

Configuration is passed explicitly (``BuildConfig``); no module global, env var or rebinding carries programme state.
"""

from memory_kernel.pipeline.config import BuildConfig
from memory_kernel.pipeline.runner import run_build

__all__ = ["BuildConfig", "run_build"]
