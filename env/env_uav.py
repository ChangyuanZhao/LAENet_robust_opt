import gym
import numpy as np
from gym import spaces
import torch
import torch.nn.functional as F

import gym
import torch
from gym.spaces import Box, MultiDiscrete, Tuple
from tianshou.env import DummyVectorEnv
from gym import spaces
from gym.utils import seeding
from numpy import linalg as LA
import math
import numpy as np
import gym
from scipy.stats import truncnorm
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.linalg as la
from numpy.linalg import eig, norm

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
# from pyargus.antenna_array import array_planner, steered_response



class CSIEnv(gym.Env):
    """Gym environment for CSI generation and interaction."""

    def __init__(self):
        super(CSIEnv, self).__init__()

        # 固定参数
        self.c = 3e8
        self.fc = 3.2e9
        self.lambda_c = self.c / self.fc  # 波长
        self.b_b = 0.05
        self.b_e = 0.05
        self.b_a = self.lambda_c / 2  # 天线间距
        self.rho = 0.95

        self.K_b = 1e0
        self.K_e = 1e0

        self.alpha_b = 0.95
        self.alpha_e = 0.3



        self.if_robust = False

        self.num_samples = 50

        self.shape = None

        # 设备位置
        self.c_a = np.array([0, 0, 0])
        self.c_b = np.array([-100, 150, 200])
        self.c_e = np.array([-90, 150, 160])

        # 观测空间：信道矩阵 (N_b, N_x * N_y)
        self.N_x, self.N_y = 4, 4
        self.N_b, self.N_e = 6, 6
        self.t = 2
        self.state_shape = (1, self.N_b * self.N_x * self.N_y)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.state_shape, dtype=np.float32
        )

        self.action_state = (1, 2 * self.N_x * self.N_y + 2 * self.N_x * self.N_y * self.t)

        # 创建新的 action_space，范围调整为 -1 到 1
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.action_state,  # 形状应匹配 action_state
            dtype=np.float32
        )

        low = self.action_space.low  # 形状与 self.action_state 相同
        high = self.action_space.high  # 形状与 self.action_state 相同

        # print(f"low.shape: {low.shape}, high.shape: {high.shape}")

        beta_0_dB = -70
        beta_0 = 10 ** (beta_0_dB / 10)

        d_b = np.linalg.norm(self.c_a - self.c_b)

        eta_b = 3.2

        delta = 1e-12



        self.power = 10 ** (10. / 10) / (beta_0 * d_b ** (-1 * eta_b)) * delta

        # print(f"power :{self.power}")


        # 初始化环境状态
        self.reset()

    def Ray_channel(self, rows, cols):
        return (np.random.randn(rows, cols) + 1j * np.random.randn(rows, cols))/np.sqrt(2)

    def generate_CSI(self, alpha_b, alpha_e):
        """生成信道状态信息 (CSI)"""

        # 估计窃听端坐标
        noise = np.random.randn(3) * np.sqrt(1 - self.rho)
        c_e_est = (self.c_e - noise) / np.sqrt(self.rho)

        # 计算LOS路径角度
        varphi_b = -np.arctan(abs(self.c_b[0]) / abs(self.c_b[1]))
        theta_b = np.pi / 2 - np.arctan(abs(self.c_b[2]) / np.sqrt(self.c_b[0] ** 2 + self.c_b[1] ** 2))
        phi_b = np.arctan(abs(self.c_b[0]) / abs(self.c_b[1]))
        vartheta_b = theta_b;

        varphi_e = -np.arctan(abs(c_e_est[0]) / abs(c_e_est[1]))
        theta_e = np.pi / 2 - np.arctan(abs(c_e_est[2]) / np.sqrt(c_e_est[0] ** 2 + c_e_est[1] ** 2))
        phi_e = np.arctan(abs(c_e_est[0]) / abs(c_e_est[1]))
        vartheta_e = theta_e;


        beta_0_dB = -70
        beta_0 = 10 ** (beta_0_dB / 10)

        d_b = np.linalg.norm(self.c_a - self.c_b)
        d_e = np.linalg.norm(self.c_a - c_e_est)

        eta_b = 3.2
        eta_e = 3.2
        delta_b = np.sqrt(1e-12)
        delta_e = np.sqrt(1e-12)

        h_bLA = np.exp(-1j * 2 * np.pi / self.lambda_c * self.b_b * np.cos(phi_b) * np.sin(vartheta_b) * np.arange(self.N_b))
        h_eLA = np.exp(-1j * 2 * np.pi / self.lambda_c * self.b_e * np.cos(phi_e) * np.sin(vartheta_e) * np.arange(self.N_e))

        H_temp1 = np.tile(np.arange(self.N_x)[:, np.newaxis], (1, self.N_y))
        H_temp2 = np.tile(np.arange(self.N_y)[np.newaxis, :], (self.N_x, 1))

        H_bLD = np.exp(-1j * 2 * np.pi / self.lambda_c * self.b_a * np.sin(theta_b) * (
                    np.cos(varphi_b) * H_temp1 - np.sin(varphi_b) * H_temp2))
        H_eLD = np.exp(-1j * 2 * np.pi / self.lambda_c * self.b_a * np.sin(theta_e) * (
                    np.cos(varphi_e) * H_temp1 - np.sin(varphi_e) * H_temp2))

        h_bLD = H_bLD.reshape(self.N_x * self.N_y, 1).T
        h_eLD = H_eLD.reshape(self.N_x * self.N_y, 1).T

        H_bL = h_bLA[:, np.newaxis] * h_bLD
        H_eL = h_eLA[:, np.newaxis] * h_eLD


        H_bk = np.zeros((self.N_b, self.N_x * self.N_y), dtype=complex)
        H_ek = np.zeros((self.N_b, self.N_x * self.N_y), dtype=complex)
        H_bt = np.zeros((self.N_b, self.N_x * self.N_y), dtype=complex)
        H_et = np.zeros((self.N_b, self.N_x * self.N_y), dtype=complex)

        h_H_bN = self.Ray_channel(self.N_b, self.N_x * self.N_y)
        d_H_bN = self.Ray_channel(self.N_b, self.N_x * self.N_y)
        H_bN = np.sqrt(alpha_b) * h_H_bN + np.sqrt(1 - alpha_b) * d_H_bN

        h_H_eN = self.Ray_channel(self.N_e, self.N_x * self.N_y)
        d_H_eN = self.Ray_channel(self.N_e, self.N_x * self.N_y)
        H_eN = np.sqrt(alpha_e) * h_H_eN + np.sqrt(1 - alpha_e) * d_H_eN

        H_bk[ :, :] = np.sqrt(self.K_b / (1 + self.K_b)) * H_bL + np.sqrt(1 / (1 + self.K_b)) * h_H_bN
        H_ek[ :, :] = np.sqrt(self.K_e / (1 + self.K_e)) * H_eL + np.sqrt(1 / (1 + self.K_e)) * h_H_eN
        H_bt[ :, :] = np.sqrt(self.K_b / (1 + self.K_b)) * H_bL + np.sqrt(1 / (1 + self.K_b)) * H_bN
        H_et[ :, :] = np.sqrt(self.K_e / (1 + self.K_e)) * H_eL + np.sqrt(1 / (1 + self.K_e)) * H_eN

        return H_bk, H_ek, H_bt, H_et

    def reset(self):
        """重置环境"""
        self.alpha_b = 0.95
        self.alpha_e = 0.3
        # H_bk, H_ek, H_bt, H_et = self.generate_CSI(self.alpha_b, self.alpha_e)
        #
        # H_b = np.concatenate([np.real(H_bk), np.imag(H_bk)], 1)
        # H_e = np.concatenate([np.real(H_ek), np.imag(H_ek)], 1)
        # H_input = np.concatenate([H_b, H_e], 1)  # for input to the network
        #
        # print(f"H shape {H_input.reshape(-1).shape}")

        # self.shape = H_input.shape

        # print(self.shape)

        H_input = np.zeros(self.state_shape) + np.random.normal(0, 0.01, size=self.state_shape)

        # print(f"h_input shape {H_input.shape}")

        return H_input

    def step(self, action):
        """执行动作"""
        temp = [action, self.power]

        # print(action.shape)

        f, G1 = self.f_G_and_power(temp)

        c_a = np.array([[0], [0], [0]])
        c_b = np.array([[-100], [150], [200]])
        c_e = np.array([[-90], [150], [160]])
        # c_e = io.loadmat('./c_e/c_e1.mat')['c__e']
        beta_0_dB = -70
        beta_0 = 10 ** (beta_0_dB / 10)
        eta_b, eta_e = 3.2, 3.2
        d_b, d_e = np.linalg.norm(c_a - c_b), np.linalg.norm(c_a - c_e)

        snr_b = beta_0 * d_b ** (-1 * eta_b)
        #snr_b = np.expand_dims(np.repeat(snr_b, N), -1)
        snr_e = beta_0 * d_e ** (-1 * eta_e)
        #snr_e = np.expand_dims(np.repeat(snr_e, N), -1)
        delta = 1e-12
        #delta_ = np.expand_dims(np.repeat(1e-12, N), -1)

        rewards = []
        num_ex = 0
        p_sum = []

        for i in range(self.num_samples):
            H_bk, H_ek, H_bt, H_et = self.generate_CSI(self.alpha_b, self.alpha_e)

            reward, flag = self.Reward_calculating([f, G1, H_bt, H_et, snr_b, snr_e, delta])

            if flag > 0:
                num_ex += 1
                p_sum.append(0.)
            else:
                p_sum.append(flag * 100.)

            # reward = 0
            rewards.append(reward)

        chance_p = num_ex/self.num_samples

        if len(rewards) != len(p_sum):
            raise ValueError("两个列表长度不相等")

        result = [a + b for a, b in zip(rewards, p_sum)]
        avg_reward = np.mean(result)

        if chance_p < 0.7:
            if len(rewards) != len(p_sum):
                raise ValueError("两个列表长度不相等")

            rewards = [a + b for a, b in zip(rewards, p_sum)]


        chance_reward = np.mean(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        # chance_p = self.chance_constant(rewards, t = 50.)
        #
        # chance_reward = 0
        #
        # if chance_p <= 0:
        #     chance_reward -= (50. - max_reward)
        # else:
        #     chance_reward += chance_p * 100.


        obs = np.zeros(self.state_shape) + np.random.normal(0, 0.01, size=self.state_shape)

        # print(f"obs.shape   {obs.shape}")

        done = True  # 强化学习为持续性任务

        # self.if_robust = True

        if self.if_robust:
            return obs, min_reward, done, {}
        else:
            # return obs, avg_reward, done, {'f_forming': f, "min_reward": min_reward, "avg_reward": avg_reward, "chance_reward": chance_reward} # min
            return obs, min_reward, done, {'f_forming': f, "min_reward": min_reward, "avg_reward": avg_reward,  "chance_reward": chance_reward} # robust


    def render(self, mode="human"):
        """可视化环境（这里可以添加更具体的绘图）"""
        print("Current CSI State:")
        print(self.H_b)

    def chance_constant(self, rewards, t = 100.):
        count_greater = sum(1 for r in rewards if r > t)
        total_count = len(rewards)
        return count_greater / total_count if total_count > 0 else 0


    def f_G_and_power(self, temp):
        t = 2
        f_G_temp, P_a0 = temp
        P_a0 = P_a0  # 取出 batch 维度的 P_a0

        f_G_temp = f_G_temp.reshape(1,-1)

        # 创建一个与 f_G_temp 形状相同的全 1 数组
        # f_G_temp = np.ones_like(f_G_temp)  # 或者 np.ones(f_G_temp.shape)

        def l2_normalize_np(x, axis=1, epsilon=1e-10):
            l2_norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
            normalized = x / (l2_norm + epsilon)
            return normalized

        # print(f_G_temp.shape)
        f_G_temp = l2_normalize_np(f_G_temp)

        # print(np.sqrt(np.sum(np.square(f_G_temp), axis=1, keepdims=True)) )

        # 乘以功率缩放因子
        f_G_temp = np.sqrt(P_a0) * f_G_temp

        # 拆分 `f_G_temp`
        f_0_real, f_0_imag = f_G_temp[:, :self.N_x * self.N_y], f_G_temp[:, self.N_x * self.N_y:2 * self.N_x * self.N_y]
        G_0_real = f_G_temp[:, 2 * self.N_x * self.N_y:2 * self.N_x * self.N_y + self.N_x * self.N_y * t]
        G_0_imag = f_G_temp[:, 2 * self.N_x * self.N_y + self.N_x * self.N_y * t:2 * self.N_x * self.N_y + 2 * self.N_x * self.N_y * t]

        # 复数化
        f = f_0_real + 1j * f_0_imag
        G = G_0_real + 1j * G_0_imag

        # print(G.shape)

        G = np.array(G)

        f = np.array(f)

        # # Result
        # norm_f_squared = np.linalg.norm(f) ** 2
        # norm_G_F_squared = np.linalg.norm(G, 'fro') ** 2
        #
        # result = norm_f_squared + norm_G_F_squared
        # print("Result:", result)
        # print("minuse", P_a0-result)

        # 调整 G 的形状
        G1 = G.reshape(self.N_x * self.N_y, t)  # 变为 (batch_size, N_x*N_y, t)
        f = f.reshape(self.N_x * self.N_y, 1)

        return f, G1

    import torch

    def Reward_calculating(self, temp, alpha = 0.7):
        f, G, H_bt, H_et, snrb, snre, delta = temp
        # print(f"snrb {snrb}")
        snrb = snrb
        snre = snre
        delta = delta

        # 计算合法用户信号增益
        aa = np.matmul(H_bt, f)


        aa1 = np.matmul(aa, np.conjugate(aa).transpose(1,0))


        bb = np.matmul(H_bt, G)
        bb1 = np.matmul(bb, np.conjugate(bb.transpose(1,0)))

        # 计算合法用户的干扰 + 噪声协方差矩阵
        K_nb = snrb* bb1 + delta * np.eye(bb1.shape[1], dtype=np.complex128)
        tempb = snrb * np.matmul(aa1, np.linalg.inv(K_nb))

        # print(f"f shape, {f.shape}")
        #
        # print(f"H_et shape, {H_et.shape}")

        # 计算窃听者信号增益
        aae = np.matmul(H_et, f)
        aae1 = np.matmul(aae, np.conjugate(aae.transpose(1, 0)))

        bbe = np.matmul(H_et, G)
        bbe1 = np.matmul(bbe, np.conjugate(bbe.transpose(1, 0)))

        # 计算窃听者的干扰 + 噪声协方差矩阵
        K_ne = snre * bbe1 + delta * np.eye(bbe1.shape[1], dtype=np.complex128)
        tempe = snre * np.matmul(aae1, np.linalg.inv(K_ne))

        # 计算合法用户和窃听者的速率
        R_sb = np.log2(np.linalg.det(np.eye(tempb.shape[1], dtype=np.complex128) + tempb).real)
        R_se = np.log2(np.linalg.det(np.eye(tempe.shape[1], dtype=np.complex128) + tempe).real)

        # print(R_se)

        # 计算奖励 (Reward)
        Reward = (R_sb - R_se)

        flag = 3.0 - R_se

        return Reward * 100., flag

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)


def make_csi_env(training_num=0, test_num=0):
    """Wrapper function for MmWaveDBSBeamformingEnv.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = CSIEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([lambda: CSIEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv([lambda: CSIEnv() for _ in range(test_num)])
        test_envs.seed(0)

    return env, train_envs, test_envs



# 测试环境
if __name__ == "__main__":
    env = CSIEnv()
    state = env.reset()

    for _ in range(10):
        action = env.action_space.sample()  # 采样随机动作
        next_state, reward, done, _ = env.step(action)
        print(f"Action shape: {action.shape}  Action: {action}, Reward: {reward}")
