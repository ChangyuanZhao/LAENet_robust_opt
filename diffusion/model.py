import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
from tianshou.data import Batch, ReplayBuffer, to_torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, time, state):
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        processed_state = processed_state.squeeze(1)
        x = torch.cat([x, t, processed_state], dim=1)
        x = self.mid_layer(x)

        return x


class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        # _act = nn.Mish if activation == 'mish' else nn.ReLU
        _act = nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.q1_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
    # def forward(self, obs):
    #     return self.q1_net(obs), self.q2_net(obs)
    #
    # def q_min(self, obs):
    #     return torch.min(*self.forward(obs))
    def forward(self, state, action):
        processed_state = self.state_mlp(state)
        processed_state = processed_state.squeeze(1)

        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)

    def q_min(self, obs, action):
        # obs = to_torch(obs, device='cuda:0', dtype=torch.float32)
        # action = to_torch(action, device='cuda:0', dtype=torch.float32)
        return torch.min(*self.forward(obs, action))


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Embedding(6, embed_dim)  # 假设时间最大值为1000

    def forward(self, time):
        return self.embedding(time)


class TransformerBlock(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim=16, num_heads=4, ff_hidden_dim=256, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.time_embedding = TimeEmbedding(t_dim)

        # Embed dimension is the concatenated dimension of state, action, and time embeddings
        self.embed_dim = state_dim + action_dim + t_dim

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # Feedforward network to project to action_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, action_dim)  # Output dimension is action_dim
        )
        self.norm2 = nn.LayerNorm(action_dim)  # Normalize action_dim output

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time, state):
        # Embed the time input
        time_emb = self.time_embedding(time)  # Add sequence dimension (seq_len, batch, embed_dim)
        state = state.squeeze(1)  # Remove extra dimension if present

        # Concatenate inputs along the feature dimension
        combined_input = torch.cat([x, time_emb, state], dim=-1)  # Shape: [seq_len, batch, embed_dim]

        # Multi-head attention
        attn_output, _ = self.multi_head_attention(combined_input, combined_input, combined_input)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(combined_input + attn_output)

        # Feedforward to project to action_dim
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout(ff_output)
        out2 = self.norm2(ff_output)

        return out2



import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        # Compute gating scores
        gate_scores = self.gate(x)  # Shape: [seq_len, batch_size, num_experts]
        gate_scores = F.softmax(gate_scores, dim=-1)

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # Top-k scores and indices

        # Combine outputs from top-k experts
        moe_output = torch.zeros_like(x)
        seq_len, embed_dim = x.shape
  # Iterate over each batch
        for s in range(seq_len):  # Iterate over each sequence
            x_s = x[s, :]  # Shape: [embed_dim]
            for k in range(self.top_k):  # Process top-k experts
                expert_idx = top_k_indices[s, k].item()  # Get the expert index
                expert_score = top_k_scores[s, k]  # Get the gating score
                moe_output[s, :] += expert_score * self.experts[expert_idx](x_s)

        return moe_output



class MoETransformerBlock(nn.Module):
    def __init__(self, state_dim, action_dim, t_dim=16, num_heads=4, ff_hidden_dim=256, dropout=0.1, num_experts=4):
        super(MoETransformerBlock, self).__init__()
        self.time_embedding = TimeEmbedding(t_dim)

        # Embed dimension is the concatenated dimension of state, action, and time embeddings
        self.embed_dim = state_dim + action_dim + t_dim

        # Mixture of Experts layer
        self.moe = MoE(self.embed_dim, num_experts=num_experts)

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # Feedforward network to project to action_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, action_dim)  # Output dimension is action_dim
        )
        self.norm2 = nn.LayerNorm(action_dim)  # Normalize action_dim output

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time, state):
        # Embed the time input
        time_emb = self.time_embedding(time)  # Add sequence dimension (seq_len, batch, embed_dim)
        state = state.squeeze(1)  # Remove extra dimension if present

        # Concatenate inputs along the feature dimension
        combined_input = torch.cat([x, time_emb, state], dim=-1)  # Shape: [seq_len, batch, embed_dim]

        # Mixture of Experts layer
        moe_output = self.moe(combined_input)

        # Multi-head attention
        attn_output, _ = self.multi_head_attention(moe_output, moe_output, moe_output)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(moe_output + attn_output)

        # Feedforward to project to action_dim
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout(ff_output)
        out2 = self.norm2(ff_output)

        return out2
