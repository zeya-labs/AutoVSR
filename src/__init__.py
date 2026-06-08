"""TransferFunctionAgent package.

Graph dependencies are imported lazily so utility subpackages can be used in
lightweight environments without LangGraph installed.
"""

from .state import TransferFunctionState

__version__ = "0.1.0"
__all__ = ["create_graph", "TransferFunctionGraph", "TransferFunctionState"]


def __getattr__(name):
    if name in {"create_graph", "TransferFunctionGraph"}:
        from .graph import create_graph, TransferFunctionGraph

        return {
            "create_graph": create_graph,
            "TransferFunctionGraph": TransferFunctionGraph,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
