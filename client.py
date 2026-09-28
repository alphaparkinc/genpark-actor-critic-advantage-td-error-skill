"""Advantage Actor-Critic (A2C) Tabular Engine
100% Python Standard Library (math).
"""

import math

class TabularActorCritic:
    """Advantage Actor-Critic with softmax actor and state-value critic."""
    def __init__(self, states, actions, lr_actor=0.1, lr_critic=0.1, gamma=0.95):
        self.states = states
        self.actions = actions
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.V = {s: 0.0 for s in states}
        self.theta = {(s, a): 0.0 for s in states for a in actions}

    def policy_probs(self, state):
        logits = [self.theta[(state, a)] for a in self.actions]
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        sum_exp = sum(exps)
        return [round(e / sum_exp, 4) for e in exps]

    def update(self, state, action, reward, next_state):
        td_error = reward + self.gamma * self.V[next_state] - self.V[state]
        self.V[state] += self.lr_critic * td_error
        probs = self.policy_probs(state)
        a_idx = self.actions.index(action)
        for i, a in enumerate(self.actions):
            grad = (1.0 - probs[i]) if i == a_idx else -probs[i]
            self.theta[(state, a)] += self.lr_actor * td_error * grad

        return {
            "td_error": round(td_error, 4),
            "updated_value": round(self.V[state], 4),
            "policy_probs": dict(zip(self.actions, self.policy_probs(state)))
        }
