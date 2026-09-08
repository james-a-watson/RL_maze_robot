"""Core reinforcement-learning scaffolding.

A small, self-contained base for episodic control agents. The design keeps the
environment interaction loop in one place (``interact``) and exposes a handful
of hooks (``init``, ``step0``, ``online``, ``offline``) that subclasses fill in
with the actual learning rule. Value/action-value estimation is deliberately
left abstract so the same loop drives tabular agents, linear function
approximation (see ``rl.linear``) and neural networks (see ``rl.deep``).

Environments follow a Gym-like protocol: ``reset() -> state``,
``step(action) -> (state, reward, done, info)`` and expose ``nA`` (number of
discrete actions). Function-approximation environments additionally expose a
feature count ``nF`` and a feature matrix helper ``S_``.
"""

import random

import numpy as np


def _recent_mean(values, ep, n):
    """Mean of the most recent ``n`` recorded values up to episode ``ep``."""
    lo = max(0, ep - n + 1)
    window = values[lo:ep + 1]
    return window.mean() if len(window) else 0.0


class MRP:
    """Markov reward process agent: prediction (state-value) only.

    Runs episodes, records per-episode metrics and delegates the learning rule
    to ``online`` (per step) and/or ``offline`` (end of episode).
    """

    def __init__(self, env=None, γ=1.0, α=0.1, v0=0.0, episodes=100,
                 max_t=2000, seed=None, store=False, visual=False, view=1,
                 last=10, print_=False, **_):
        self.env = env
        self.γ = γ
        self.α = α
        self.v0 = v0
        self.episodes = episodes
        self.max_t = max_t
        self.store = store
        self.visual = visual
        self.view = view
        self.last = last
        self.print = print_

        # The policy and the per-step routine are swapped out by subclasses:
        # a Q-learning style agent only needs the current action, whereas a
        # Sarsa style agent commits to the next action in advance.
        self.policy = self.stationary
        self.step = self.step_a
        self.skipstep = False

        if env is not None:
            self.As = list(range(env.nA))
            self.pAs = [1 / env.nA] * env.nA

        self.seed(seed)
        self.ep = -1  # guards metrics access before any training has happened

    # -- metrics ---------------------------------------------------------
    def init_metrics(self):
        self.Ts = np.zeros(self.episodes, dtype=np.uint32)  # steps per episode
        self.Rs = np.zeros(self.episodes)                   # return per episode
        self.Es = np.zeros(self.episodes)                   # error per episode

    def extend_metrics(self):
        """Grow the metric buffers in place when resuming a longer run."""
        if len(self.Ts) >= self.episodes:
            return
        for name in ("Ts", "Rs", "Es"):
            getattr(self, name).resize(self.episodes, refcheck=False)

    def metrics(self):
        i = self.ep % self.episodes
        self.Ts[i] = self.t + 1
        self.Rs[i] = self.Σr
        self.Es[i] = self.Error()
        if self.print:
            print(self)

    def __str__(self):
        R = _recent_mean(self.Rs, self.ep, self.last)
        return ("step %d, episode %d, return %.2f, mean return last %d %.2f, "
                "ε %.2f" % (self.t_, self.ep, self.Σr, self.last, R,
                            getattr(self, "ε", 0)))

    # -- value estimate --------------------------------------------------
    def init_(self):
        """Allocate the value table. Overridden for function approximation."""
        self.V = np.ones(self.env.nS) * self.v0

    def V_(self, s=None):
        return self.V if s is None else self.V[s]

    def seed(self, seed=None, **_):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    # -- trajectory buffer (only used by agents that learn from batches) --
    def allocate(self):
        if not self.store:
            return
        self.r = np.zeros(self.max_t)
        self.s = np.zeros(self.max_t, dtype=np.uint32)
        self.a = np.zeros(self.max_t, dtype=np.uint32)
        self.done = np.zeros(self.max_t, dtype=bool)

    def store_(self, s=None, a=None, rn=None, sn=None, an=None, done=None, t=0):
        if not self.store:
            return
        if s is not None:
            self.s[t] = s
        if a is not None:
            self.a[t] = a
        if rn is not None:
            self.r[t + 1] = rn
        if sn is not None:
            self.s[t + 1] = sn
        if an is not None:
            self.a[t + 1] = an
        if done is not None:
            self.done[t + 1] = done

    # -- episode / experiment termination --------------------------------
    def stop_ep(self, done):
        return done or (self.t + 1 >= self.max_t - 1)

    def stop_exp(self):
        if self.stop_early():
            print("stopped early at episode %d" % self.ep)
            return True
        return self.ep >= self.episodes - 1

    # -- one interaction step, in two flavours ---------------------------
    def step_0(self):
        s = self.env.reset()
        return s, self.policy(s)

    def step_a(self, s, _, t):
        """Q-learning / value style: choose the action at the current state."""
        if self.skipstep:
            return 0, None, None, None, True
        a = self.policy(s)
        sn, rn, done, _ = self.env.step(a)
        self.store_(s=s, a=a, rn=rn, sn=sn, done=done, t=t)
        return rn, sn, a, None, done

    def step_an(self, s, a, t):
        """Sarsa style: also choose the next action, needed by the update."""
        if self.skipstep:
            return 0, None, None, None, True
        sn, rn, done, _ = self.env.step(a)
        an = self.policy(sn)
        self.store_(s=s, a=a, rn=rn, sn=sn, an=an, done=done, t=t)
        return rn, sn, a, an, done

    # -- the main loop ---------------------------------------------------
    def interact(self, train=True, resume=False, episodes=0, **kw):
        if episodes:
            self.episodes = episodes
        if train and not resume:
            self.init_()          # allocate value estimate
            self.init()           # subclass set-up before the first episode
            self.init_metrics()
            self.allocate()
            self.plot0()
            self.seed(**kw)
            self.ep = -1
            self.t_ = 0           # global step counter across all episodes
        if resume:
            self.extend_metrics()

        while not self.stop_exp():
            self.ep += 1
            self.t = -1           # per-episode step counter
            self.Σr = 0
            done = False

            s, a = self.step_0()
            self.step0()          # subclass set-up at the start of each episode

            while not self.stop_ep(done):
                self.t += 1
                self.t_ += 1

                rn, sn, a, an, done = self.step(s, a, self.t)
                if train:
                    self.online(s, rn, sn, done, a, an)

                self.Σr += rn
                self.rn = rn
                s, a = sn, an

                if self.visual and self.episodes > self.ep >= self.episodes - self.view:
                    self.render(**kw)

            self.metrics()
            if train:
                self.offline()
            self.plot_ep()

        self.plot_exp(**kw)
        return self

    # -- default (exploration-free) policy -------------------------------
    def stationary(self, *_):
        return random.choices(self.As, weights=self.pAs, k=1)[0]

    # -- hooks subclasses may override -----------------------------------
    def init(self):
        pass

    def step0(self):
        pass

    def online(self, *args):
        pass

    def offline(self):
        pass

    def Error(self):
        return 0

    def stop_early(self):
        return False

    def plot0(self):
        pass

    def plot_ep(self):
        pass

    def plot_exp(self, *args):
        pass

    def render(self, rn=None, label="", **kw):
        if rn is None:
            rn = self.rn
        self.env.render(label="%s reward=%d t=%d ep=%d"
                        % (label, rn, self.t + 1, self.ep + 1), **kw)


class MDP(MRP):
    """Markov decision process agent: ε-greedy action-value control."""

    def __init__(self, env=None, ε=0.1, εmin=0.01, dε=1.0, εT=0, q0=0.0,
                 commit_ep=0, Tstar=0, **kw):
        super().__init__(env=env, **kw)
        self.ε = ε
        self.ε0 = ε        # remembered so linear decay can reference the start
        self.εmin = εmin
        self.dε = dε       # multiplicative (exponential) decay per step
        self.εT = εT       # horizon for linear decay, in global steps
        self.q0 = q0
        self.commit_ep = commit_ep
        self.Tstar = Tstar

        self.policy = self.εgreedy

    def init_(self):
        super().init_()
        self.Q = np.ones((self.env.nS, self.env.nA)) * self.q0

    def Q_(self, s=None, a=None):
        return self.Q[s] if s is not None else self.Q

    def greedy_(self, s):
        """Deterministic greedy action. For evaluation, not for learning."""
        return int(np.argmax(self.Q_(s)))

    def greedy(self, s):
        """Greedy action with random tie-breaking among joint maxima."""
        Qs = self.Q_(s)
        best = np.where(Qs == Qs.max())[0]
        return int(random.choice(best))

    def εgreedy(self, s):
        if self.dε < 1:
            self.ε = max(self.εmin, self.ε * self.dε)
        if self.εT > 0:
            self.ε = max(self.εmin, self.ε0 - self.t_ / self.εT)
        if np.random.rand() > self.ε:
            return self.greedy(s)
        return np.random.randint(0, self.env.nA)

    def π(self, s, a=None):
        """ε-greedy action probabilities at state ``s``."""
        Qs = self.Q_(s)
        probs = np.zeros_like(Qs, dtype=float) + self.ε / self.env.nA
        probs[Qs.argmax()] += 1 - self.ε
        return probs if a is None else probs[a]

    def πisoptimal(self):
        """Whether the greedy policy reaches a terminal state within Tstar."""
        s = self.env.reset()
        done = False
        for _ in range(self.Tstar):
            s, _, done, _ = self.env.step(self.greedy_(s))
        return done
