"""Linear function approximation.

Values are linear in a feature vector ``x`` produced by the environment:

    V(s)   = w · x(s)
    Q(s,a) = W[a] · x(s)

Because the estimate is linear, the gradient with respect to the weights is
just the feature vector itself, which is why ``ΔV`` / ``ΔQ`` simply return the
state. Tile-coded (sparse, binary) features keep these dot products cheap.

The environment supplies feature vectors directly as its "state" and reports
the feature dimension as ``nF``.
"""

import numpy as np

from . import core


class MRP(core.MRP):
    """State-value prediction with a linear estimate ``V = w · x``."""

    def init_(self):
        # No value table to allocate: the estimate lives in the weights.
        pass

    def init(self):
        self.w = np.ones(self.env.nF) * self.v0
        self.V = self.V_

    def V_(self, s=None):
        return self.w.dot(s) if s is not None else self.w.dot(self.env.S_())

    def ΔV(self, s):
        return s


class MDP(core.MDP, MRP):
    """ε-greedy control with a linear action-value estimate ``Q = W · x``."""

    def init_(self):
        pass

    def init(self):
        super().init()  # sets up the critic weights w (used by actor-critic)
        self.W = np.ones((self.env.nA, self.env.nF)) * self.q0
        self.Q = self.Q_

    def Q_(self, s=None, a=None):
        W = self.W if a is None else self.W[a]
        if s is not None:
            return W.dot(s)
        return np.matmul(W, self.env.S_()).T

    def ΔQ(self, s):
        return s


class Sarsa(MDP):
    """On-policy TD control (Sarsa) with linear function approximation."""

    def init(self):
        super().init()
        self.step = self.step_an  # commit to the next action before updating

    def online(self, s, rn, sn, done, a, an):
        target = rn + (1 - done) * self.γ * self.Q(sn, an)
        self.W[a] += self.α * (target - self.Q(s, a)) * self.ΔQ(s)


class Sarsaλ(MDP):
    """Sarsa(λ): Sarsa with accumulating eligibility traces.

    A separate trace vector ``Z[a]`` is kept per action. Each step the TD error
    is applied not only to the feature just visited but, with geometrically
    decaying weight ``(γλ)^k``, to recently visited features too. This spreads
    credit backwards along the trajectory and typically speeds convergence.
    """

    def __init__(self, λ=0.5, **kw):
        super().__init__(**kw)
        self.λ = λ
        self.step = self.step_an

    def step0(self):
        self.Z = self.W * 0  # reset traces at the start of every episode

    def online(self, s, rn, sn, done, a, an):
        self.Z[a] = self.λ * self.γ * self.Z[a] + self.ΔQ(s)
        target = rn + (1 - done) * self.γ * self.Q(sn, an)
        self.W[a] += self.α * (target - self.Q(s, a)) * self.Z[a]
