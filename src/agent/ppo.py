# src/agent/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent:
    def __init__(self, actor, critic, lr=1e-4, gamma=0.99, clip_ratio=0.2, update_epochs=10):
        self.actor = actor  # GNN + Head
        self.critic = critic  # GNN or MLP
        self.optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs

        # バッファ
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        with torch.no_grad():
            logits, probs = self.actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state)

        return action, log_prob, value

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self):
        returns = []
        advantages = []
        G = 0
        A = 0
        next_value = 0

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * next_value * (1 - self.dones[step]) - self.values[step]
            A = delta + self.gamma * self.clip_ratio * A
            G = self.rewards[step] + self.gamma * G
            returns.insert(0, G)
            advantages.insert(0, A)
            next_value = self.values[step]

        return returns, advantages

    def update(self):
        returns, advantages = self.compute_returns_and_advantages()

        for _ in range(self.update_epochs):
            for state, action, old_log_prob, R, adv in zip(self.states, self.actions, self.log_probs, returns, advantages):
                logits, probs = self.actor(state)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action)
                ratio = torch.exp(log_prob - old_log_prob)

                # clipped objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(state)
                critic_loss = (R - value).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.clear_buffer()

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
