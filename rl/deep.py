"""Neural-network function approximation.

Estimates V and Q with Keras networks instead of a linear map, which lets the
raw (normalised) sensor readings be used as the state without hand-designed
features. Two ingredients make training with a network stable enough to be
usable online:

* an **experience-replay buffer**, so updates are made on batches of past
  transitions rather than only the latest, highly correlated one;
* a **target network** (``qNn``) that lags the online network and supplies the
  bootstrap estimate, so the regression target does not chase itself.

This module provides the value/action-value plumbing (network creation,
buffer, batching, prediction and fitting). Concrete algorithms subclass
``MDP`` and implement ``online`` with their own update rule and, if needed,
their own ``create_model`` architecture.
"""

from collections import deque
from itertools import islice
from random import sample

import numpy as np

try:
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
except ImportError as e:  # pragma: no cover - only hit without TensorFlow
    raise ImportError(
        "rl.deep needs TensorFlow. Install it with `pip install tensorflow` "
        "(only the deep-RL model requires it; the linear model does not)."
    ) from e

from . import core


class MRP(core.MRP):
    """State-value prediction with a neural-network estimate of V."""

    def __init__(self, γ=0.99, nF=512, nbuffer=10000, nbatch=32, rndbatch=True,
                 save_weights=1000, load_weights=False, print_=True, **kw):
        super().__init__(γ=γ, print_=print_, **kw)
        self.nF = nF            # width of the penultimate dense layer
        self.nbuffer = nbuffer  # replay-buffer capacity / minimum fill
        self.nbatch = nbatch
        self.rndbatch = rndbatch
        self.load_weights_ = load_weights
        self.save_weights_ = save_weights

    def init(self):
        self.vN = self.create_model("V")
        if self.load_weights_:
            self.load_weights(self.vN, "V")
        self.V = self.V_

    # -- network construction -------------------------------------------
    def create_model(self, net_str):
        x0 = Input(self.env.reset().shape)
        x = Conv2D(32, 8, 4, activation="relu")(x0)
        x = Conv2D(64, 4, 2, activation="relu")(x)
        x = Conv2D(64, 3, 1, activation="relu")(x)
        x = Flatten()(x)
        x = Dense(self.nF, activation="relu")(x)
        x = Dense(1 if net_str == "V" else self.env.nA)(x)
        model = Model(x0, x)
        model.compile(Adam(self.α), loss="mse")
        model.net_str = net_str
        return model

    def load_weights(self, net, net_str):
        net.load_weights(net_str).assert_consumed()

    def save_weights(self):
        self.vN.save_weights("V")

    # -- value estimate: predict when no target is passed, else fit ------
    def V_(self, s, Vs=None):
        if Vs is not None:
            self.vN.fit(s, Vs, verbose=0)
            return None
        # Single state (add a batch axis) versus an already-batched tensor.
        if len(s.shape) != 4:
            return self.vN.predict(np.expand_dims(s, 0))[0]
        return np.copy(self.vN.predict(s))

    # -- replay buffer ---------------------------------------------------
    def allocate(self):
        self.buffer = deque(maxlen=self.nbuffer)

    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        if self.save_weights_ and self.t_ % self.save_weights_ == 0:
            self.save_weights()
        self.buffer.append((s, a, rn, sn, done))

    def slice_(self, buffer, nbatch):
        # deque does not support slicing, so take the last nbatch via islice.
        return list(islice(buffer, len(buffer) - nbatch, len(buffer)))

    def batch(self):
        samples = (sample(self.buffer, self.nbatch) if self.rndbatch
                   else self.slice_(self.buffer, self.nbatch))
        # Transpose the list of 5-tuples into 5 batched arrays.
        samples = [np.array(col) for col in zip(*samples)]
        inds = np.arange(self.nbatch)
        return samples, inds


class MDP(core.MDP, MRP):
    """ε-greedy control with a neural-network estimate of Q and a target net."""

    def __init__(self, create_vN=False, **kw):
        super().__init__(**kw)
        self.create_vN = create_vN  # actor-critic also needs a value network

    def init(self):
        if self.create_vN:
            super().init()                 # build the value network vN
        self.qN = self.create_model("Q")   # online action-value network
        self.qNn = self.create_model("Qn")  # target network for bootstrapping
        if self.load_weights_:
            self.load_weights(self.qN, "Q")
            self.load_weights(self.qNn, "Q")
        self.Q = self.Q_

    def save_weights(self):
        if self.create_vN:
            super().save_weights()
        self.qN.save_weights("Q")

    def set_weights(self, net):
        net.set_weights(self.qN.get_weights())

    def Q_(self, s, Qs=None):
        if Qs is not None:
            self.qN.fit(s, Qs, verbose=0)
            return None
        if len(s.shape) != 4:
            return self.qN.predict(np.expand_dims(s, 0))[0]
        return np.copy(self.qN.predict(s))

    def Qn(self, sn, update=False):
        if update:
            self.set_weights(self.qNn)
            return None
        return self.qNn.predict(sn)
