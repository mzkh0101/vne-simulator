# src/agent/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim


class PPOAgent:
    def __init__(self, actor_encoder, actor_head, critic_encoder, critic_head, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # モデル
        self.actor_encoder = actor_encoder.to(self.device)
        self.actor_head = actor_head.to(self.device)
        self.critic_encoder = critic_encoder.to(self.device)
        self.critic_head = critic_head.to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor_encoder.parameters()) +
            list(self.actor_head.parameters()) +
            list(self.critic_encoder.parameters()) +
            list(self.critic_head.parameters()),
            lr=config["lr"]
        )

        self.gamma = config["gamma"]
        self.clip_ratio = config["clip_ratio"]
        self.update_epochs = config["update_epochs"]

        self.clear_buffer()

    def select_action(self, vnr_data):
        vnr_data = vnr_data.to(self.device)
        with torch.no_grad():
            features = self.actor_encoder(vnr_data)
            logits, probs = self.actor_head(features)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value_feat = self.critic_encoder(vnr_data)
            value = self.critic_head(value_feat).squeeze()

        return action, log_prob, value

    def store_transition(self, vnr_data, action, log_prob, reward, done, value):
        self.states.append(vnr_data)
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
                state = state.to(self.device)
                features = self.actor_encoder(state)
                logits, probs = self.actor_head(features)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action)
                ratio = torch.exp(log_prob - old_log_prob)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_feat = self.critic_encoder(state)
                value = self.critic_head(value_feat).squeeze()
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
