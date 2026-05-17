"""hprobes — Hallucination neuron discovery and causal validation for transformer LLMs."""

from importlib.metadata import version as _metadata_version

from .probe import HProbes

__all__ = ["HProbes"]
__version__ = _metadata_version("hprobes")
