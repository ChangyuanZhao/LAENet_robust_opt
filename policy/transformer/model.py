import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        return self.transformer(x).squeeze(1)

class TransformerActor(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim=256, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        
        self.transformer = TransformerEncoder(
            input_dim=state_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        x = self.transformer(state)
        return self.action_head(x)

class DoubleCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        def create_critic():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        
        self.q1 = create_critic()
        self.q2 = create_critic()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
class TransformerPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        model,
        max_action,
        bc_coef=False
    ):
        super().__init__()
        self.model = model
        self.max_action = max_action
        self.bc_coef = bc_coef
        
    def forward(self, state):
        action = self.model(state)
        return self.max_action * action  # Scale to max action range
    
    def loss(self, expert_actions, states):
        pred_actions = self.forward(states)
        return torch.mean((pred_actions - expert_actions) ** 2)