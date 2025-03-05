import numpy as np
N = 100000
t = 2  # (,*) dimension of G
# parameters
N_x, N_y, N_b, N_e = 4, 4, 6, 6
c_a = np.array([[0], [0], [0]])
c_b = np.array([[-100], [150], [200]])
c_e = np.array([[100], [150], [220]])
# c_e = io.loadmat('./c_e/c_e1.mat')['c__e']
beta_0_dB = -70
beta_0 = 10**(beta_0_dB/10)
eta_b, eta_e = 3.2, 3.2
d_b, d_e = np.linalg.norm(c_a-c_b), np.linalg.norm(c_a-c_e)
snr_b = beta_0*d_b**(-1*eta_b)
snr_b = np.expand_dims(np.repeat(snr_b, N), -1)
snr_e = beta_0*d_e**(-1*eta_e)
snr_e = np.expand_dims(np.repeat(snr_e, N), -1)
delta_ = np.expand_dims(np.repeat(1e-12, N), -1)

print(snr_b)


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


