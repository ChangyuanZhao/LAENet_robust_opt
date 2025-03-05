import gym
import torch
from gym.spaces import Box, MultiDiscrete, Tuple
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np
from gym import spaces
from gym.utils import seeding
from numpy import linalg as LA
import math
import numpy as np
import gym
from scipy.stats import truncnorm

class AIGCEnv(gym.Env):

    def __init__(self):
        self._num_integer_actions = 8
        self._num_float_actions = 4
        self._num_env_floats = 4
        self._num_tasks = 4
        self._flag = 0
        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self.last_expert_action = None
        self._last_action = None
        self._steps_per_episode = 1

        # Define action space
        self._action_space = MultiDiscrete([100] * self._num_integer_actions)


        # Define observation space: the last action (5 integers + 4 floats) + 9 environment floats
        self._observation_space = Box(low=0, high=100, shape=(25,), dtype=np.float32)

    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    @property
    def action_space(self):
        # Return the action space
        return self._action_space


    @property
    def state(self):
        # Generate random channel gain
        #np.random.gauss(1e-3, 1e-6, self._num_env_floats)
        #np rarom gausians
        env_floats = np.random.uniform(1e-3, 1e-6,self._num_env_floats)
        
        # If it's the first step, initialize the last action randomly
        if self._laststate is None:
            integer_actions = np.random.randint(1, 101, self._num_integer_actions)
            power = np.random.randint(1, 101, 4)
            compress = np.random.randint(1, 10, 4)
            quality =np.random.randint(0, 1, 4)
            reward = sum(quality)/4
            reward = np.array([reward])
            state = np.concatenate([integer_actions,env_floats,power,compress,quality,reward])
        else:
            integer_actions = self._last_action
            state = self._laststate
        print(f"State shape: {state.shape}")
        # Concatenate the last action with the environment floats
        
        return state


    def step(self, action):
        # Check if episode has ended
        assert not self._terminated, "One episodic has terminated"
        self._last_action = action
        # channel_gains = self._laststate[self._num_integer_actions:self._num_integer_actions+self._num_env_floats]
        channel_gains = np.random.uniform(1e-6, 9e-7, self._num_env_floats)
        #check if last 4 actions are more than 1
        if sum(action[4:8]) > 100:
            self._terminated = True
            return self._laststate, -10, self._terminated,{"quality":0}
        # Calculate reward based on last state and action taken
        reward, real_action,power,comrpess,quality= CompUtility(channel_gains, action)
        reward = np.array([reward])
        self._laststate = np.concatenate([real_action, channel_gains,power, comrpess, quality, reward])
        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True
        # Information about number of steps taken
        print(f"State shape after step: {self._laststate.shape}")
        return self._laststate, reward, self._terminated,{"quality":quality}

    def reset(self):
        # Reset the environment to its initial state
        self._num_steps = 0
        self._terminated = False
        self._last_action = np.random.randint(1, 101, self._num_integer_actions)  # Initialize _last_action
        integer_actions = np.random.randint(1, 101, self._num_integer_actions)
        power = np.random.randint(1, 101, 4)
        compress = np.random.randint(1, 10, 4)
        quality = np.random.randint(0, 1, 4)
        reward = sum(quality) / 4
        reward = np.array([reward])
        env_floats = np.zeros(self._num_env_floats)  # Assuming env_floats is defined somewhere
        self._laststate = np.concatenate([integer_actions, env_floats, power, compress, quality, reward])
        state = self._laststate
        print(f"State shape: {state.shape}")
        return state

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)


def make_aigc_env(training_num=0, test_num=0):
    """Wrapper function for AIGC env.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs


def make_mmwave_env(training_num=0, test_num=0):
    """Wrapper function for MmWaveDBSBeamformingEnv.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = MmWaveDBSBeamformingEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv([lambda: MmWaveDBSBeamformingEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv([lambda: MmWaveDBSBeamformingEnv() for _ in range(test_num)])
        test_envs.seed(0)

    return env, train_envs, test_envs


class MmWaveDBSBeamformingEnv(gym.Env):
    def __init__(self, N=500, M=20, K=100, hd=50.0, gamma=10, noise_power=1e-9, num_samples=10, csi_noise_std=0.1,
                 uav_variation_std=1.0, if_robust = False, lambda_gb=0.2, hb = 1.75, hr = 1.65, t_power = 1, f_c=6):
        super(MmWaveDBSBeamformingEnv, self).__init__()

        # whether robust optimization
        self.if_robust = if_robust
        self.f_c = f_c
        self.t_power = t_power

        self.hb = hb
        self.hr = hr

        # 用户数
        self.N = N
        # 天线阵列数
        self.M = M
        # 可选波束角数
        self.K = K
        # 无人机基站高度
        self.hd = hd
        # SNR 阈值
        self.gamma = gamma
        # 噪声功率
        self.noise_power = noise_power
        # 采样次数
        self.num_samples = num_samples
        # 信道估计误差标准差
        self.csi_noise_std = csi_noise_std
        # UAV 位置小范围波动
        self.uav_variation_std = uav_variation_std

        self.lambda_gb = lambda_gb

        # 波束角范围 (-90° to 90°) 离散化
        self.beam_angles = np.linspace(-np.pi / 2, np.pi / 2, K)
        self.action_space = spaces.Discrete(K)

        # 观测空间：当前波束角
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 初始化 UAV 位置
        self.uav_position = np.array([50.0, 50.0, self.hd])

        self.use_users = self._generate_users()

    def _truncated_normal(self, mean, std, lower, upper):
        """ 生成截断高斯分布的随机数 """
        return truncnorm.rvs((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)

    def _generate_users(self):
        """ 生成用户的随机位置 """
        users = []
        for _ in range(self.N):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            z = 1.5  # 用户设备高度假设
            users.append([x, y, z])
        return users

    def _calculate_angles(self):
        """ 在 step 中计算 UAV 位置的变化，并重新计算角度 """
        self.uav_position += np.clip(np.random.normal(0, self.uav_variation_std, 3), -3*self.uav_variation_std, 3*self.uav_variation_std)  # UAV 位置小范围波动
        self.hd = self.uav_position[2]
        angles = []
        for user in self.users:
            dx, dy, dz = user[0] - self.uav_position[0], user[1] - self.uav_position[1], user[2] - self.uav_position[2]
            azimuth = np.arctan2(dy, dx)
            elevation = np.arctan2(np.abs(dz), np.sqrt(dx ** 2 + dy ** 2))
            # print(elevation)
            azimuth_with_error = self._truncated_normal(azimuth, np.pi / 36, -np.pi / 2, np.pi / 2)
            elevation_with_error = self._truncated_normal(elevation, np.pi / 36, -np.pi / 2, np.pi / 2)

            # p_los = 1-np.exp(-self.lambda_gb * np.sqrt(dx ** 2 + dy ** 2) * (self.hb - self.hr) / (self.hd - self.hr))

            # print(np.sqrt(dx ** 2 + dy ** 2))

            p_los, _ = self._los_nlos_probability(np.sqrt(dx ** 2 + dy ** 2))

            angles.append((azimuth_with_error, elevation_with_error, p_los, np.sqrt(dx ** 2 + dy ** 2 + dz**2)))

            # print(elevation_with_error/(np.pi/2))

        return angles

    def _los_nlos_probability(self, d, a=9.61, b=0.16):
        """
        计算 LoS 和 NLoS 概率

        参数:
            d : float 或 array
                UAV 与地面用户的水平距离 (m)
            a : float, 默认 9.61
                LoS 概率计算的环境参数
            b : float, 默认 0.16
                LoS 概率计算的环境参数

        返回:
            P_LoS : float 或 array
                视距 (LoS) 概率
            P_NLoS : float 或 array
                非视距 (NLoS) 概率
        """
        P_LoS = 1 / (1 + a * np.exp(-b * (d - a)))
        P_NLoS = 1 - P_LoS
        return P_LoS, P_NLoS

    def _calculate_snr(self, phi, angles):
        """ 计算当前波束角下的用户 SNR，并加入 CSI 估计误差 """
        covered_users = 0
        for (azimuth, elevation, p_los, d) in angles:
            # print(p_los)
            # 3GPP TR 38.901 NLOS 路径损耗模型
            sigma_shadowing = np.random.normal(0, 8)  # NLOS 下的阴影衰落 (通常 6-12 dB)

            sigma_shadowing = np.clip(sigma_shadowing, -3*8, 3*8)  # 限制范围

            path_loss_dB = 40.05 + 20 * np.log10(d) + 20 * np.log10(self.f_c) + sigma_shadowing
            path_loss = 10 ** (path_loss_dB / 10)  # 线性值

            # 3GPP TR 38.901 LOS 路径损耗模型
            sigma_shadowing = np.random.normal(0, 4)  # NLOS 下的阴影衰落 (通常 6-12 dB)

            sigma_shadowing = np.clip(sigma_shadowing, -3 * 4, 3 * 4)  # 限制范围

            path_loss_dB = 32.4 + 20 * np.log10(d) + 31.9 * np.log10(self.f_c) + sigma_shadowing
            path_loss_los = 10 ** (path_loss_dB / 10)  # 线性值

            # path_loss = (self.hd / (np.cos(elevation) + 1e-6)) ** 2.5
            # non_path_loss = (self.hd / (np.cos(elevation) + 1e-6)) ** 2.5
            steering_vector = np.exp(-1j * np.pi * np.arange(self.M) * np.sin(elevation))
            beamforming_vector = np.exp(-1j * np.pi * np.arange(self.M) * np.sin(phi)) / np.sqrt(self.M)
            array_gain = np.abs(np.dot(steering_vector.conj().T, beamforming_vector)) ** 2

            #if fading_type == 'rayleigh':
            h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)  # 标准瑞利分布

            # 计算 SNR (线性值)
            snr = (self.t_power * np.abs(h) ** 2 * array_gain) / (path_loss * self.noise_power) * (1-p_los) + (self.t_power * np.abs(h) ** 2 * array_gain) / (path_loss_los * self.noise_power) * (p_los)

            snr_with_noise = snr * (1 + np.clip(np.random.normal(0, self.csi_noise_std), -3*self.csi_noise_std, 3*self.csi_noise_std))  # 添加 CSI 估计误差
            if snr_with_noise >= self.gamma:
                covered_users += 1
        return covered_users

    def reset(self):
        """ 重置环境 """
        #self.users = self.use_users
        self.users = self._generate_users()
        self.uav_position = np.array([50.0, 50.0, self.hd])
        self.current_phi = self.beam_angles[0] # np.random.choice(self.beam_angles)
        return np.array([self.current_phi / (np.pi / 2)], dtype=np.float32)

    def step(self, action):
        # print(action)
        """ 执行波束选择，并计算多个样本的奖励均值和最小值 """
        # action = np.random.rand(*action.shape)

        # action_index = np.argmax(action)  # 选择概率最高的角度
        #
        # print(action_index)
        print(action)
        # #
        # action_index = 87

        # action_index = np.random.randint(0, 500)


        self.current_phi = action * (np.pi/2)
        angles = self._calculate_angles()
        rewards = [self._calculate_snr(self.current_phi, angles) for _ in range(self.num_samples)]
        avg_reward = np.mean(rewards)
        min_reward = np.min(rewards)
        obs = np.array([self.current_phi / (np.pi / 2)], dtype=np.float32)

        done = True  # 强化学习为持续性任务

        # self.if_robust = True

        if self.if_robust:
            return obs, min_reward, done, {}
        else:
            return obs, avg_reward, done, {}

    def render(self, mode='human'):
        print(f"Current UAV Position: {self.uav_position}, Current Beam Angle: {self.current_phi * 180 / np.pi:.2f}°")

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)
