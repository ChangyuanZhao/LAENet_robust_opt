import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR

class TransformerOPT(BasePolicy):
    def __init__(
            self,
            state_dim: int,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            critic: Optional[torch.nn.Module],
            critic_optim: Optional[torch.optim.Optimizer],
            device: torch.device,
            tau: float = 0.005,
            gamma: float = 1,
            reward_normalization: bool = False,
            estimation_step: int = 1,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            bc_coef: bool = False,
            exploration_noise: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize networks and optimizers
        if actor is not None and actor_optim is not None:
            self._actor = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim = actor_optim
            self._action_dim = action_dim

        if critic is not None and critic_optim is not None:
            self._critic = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._bc_coef = bc_coef
        self._device = device
        self.noise_generator = GaussianNoise(sigma=exploration_noise)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]
        next_actions = self(batch, model='_target_actor', input='obs_next').act
        batch.obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)
        target_q = self._target_critic.q_min(batch.obs_next, next_actions)
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        return self.compute_nstep_return(
            batch, buffer, indices, self._target_q,
            self._gamma, self._n_step, self._rew_norm
        )

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        obs = to_torch(batch[input], device=self._device, dtype=torch.float32)
        model = self._actor if model == "actor" else self._target_actor
        actions = model(obs)

        if not self._bc_coef and np.random.rand() < 0.1:
            noise = to_torch(
                self.noise_generator.generate(actions.shape),
                dtype=torch.float32,
                device=self._device
            )
            actions = torch.clamp(actions + noise, -1, 1)

        return Batch(logits=actions, act=actions, state=obs, dist=None)

    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts = to_torch(batch.act, device=self._device, dtype=torch.float32)
        current_q1, current_q2 = self._critic(obs, acts)
        target_q = batch.returns
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        return critic_loss

    def _update_actor(self, batch: Batch) -> torch.Tensor:
        obs = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        actions = self._actor(obs)
        q_value = self._critic.q_min(obs, actions)
        actor_loss = -q_value.mean()

        if self._bc_coef:
            expert_actions = torch.Tensor([info["expert_action"] for info in batch.info]).to(self._device)
            bc_loss = F.mse_loss(actions, expert_actions)
            actor_loss = bc_loss

        return actor_loss

    def _update_targets(self):
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        critic_loss = self._update_critic(batch)
        actor_loss = self._update_actor(batch)

        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()

        self._update_targets()

        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()

        return {
            'loss/critic': critic_loss.item(),
            'loss/actor': actor_loss.item()
        }

class GaussianNoise:
    def __init__(self, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        return np.random.normal(self.mu, self.sigma, shape)