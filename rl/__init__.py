"""A compact reinforcement-learning toolkit.

Submodules:
    core    - the environment-interaction loop and ε-greedy control base.
    linear  - linear function approximation (Sarsa, Sarsa(λ)).
    deep    - neural-network function approximation (needs TensorFlow).

``core`` and ``linear`` are imported eagerly; ``deep`` is left out so the
common case does not pull in TensorFlow. Import it explicitly when needed:

    from rl import deep
"""

from . import core, linear

__all__ = ["core", "linear"]
