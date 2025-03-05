import argparse
import sys
import os
import pprint
import torch
import torch.nn as nn
import numpy as np
from os import path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.policy import SACPolicy
# from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.trainer import offpolicy_trainer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from env import make_aigc_env, make_mmwave_env, make_csi_env
import scipy.io as sio
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()

    # common
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reward-threshold', type=float, default=None)
    parser.add_argument('--buffer-size', type=int, default=1)#1e6
    parser.add_argument('--epoch', type=int, default=1e6)
    parser.add_argument('--step-per-epoch', type=int, default=1)
    parser.add_argument('--step-per-collect', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--update-per-step', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=16)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.01)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)

    # for sac
    parser.add_argument('--actor-lr', type=float, default=1e-4) #1e-4
    parser.add_argument('--critic-lr', type=float, default=1e-4) #1e-3
    parser.add_argument('--alpha-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--auto-alpha', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environment
    # env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    env, train_envs, test_envs = make_csi_env(args.training_num, args.test_num)
    args.state_shape = 96

    args.action_shape = 384
    args.max_action = 1.

    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, "sac", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)


    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))


    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False

    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # policy
    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        args.alpha,
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # trainer
    if not args.watch:
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False
        )
        pprint.pprint(result)

        name = 'min'

        test_collector.reset_buffer()  # 清空历史数据

        result = test_collector.collect(n_episode=args.test_num, render=False)

        # 找到 avg_reward 最大的 buffer 索引
        best_idx = np.argmax([buf.info['avg_reward'] for buf in test_collector.buffer])

        # 获取对应的 f_forming 值
        f = test_collector.buffer[best_idx].info['f_forming']

        # 转换为 double complex 类型
        f = np.array(f, dtype=np.complex128)

        # 保存到 .mat 文件
        sio.savemat(f'./result/f_trained_sac_{name}.mat', {'f': f})

        # 存储数据的列表
        data = []

        # 遍历 test_collector.buffer
        for i in range(len(test_collector.buffer)):
            info = test_collector.buffer[i].info
            data.append({
                "min_reward": info["min_reward"],
                "avg_reward": info["avg_reward"],
                "chance_reward": info["chance_reward"],
            })

        # 转换为 DataFrame
        df = pd.DataFrame(data)

        # 存储为 Excel 文件
        excel_path = f"./result/sac_mean_{name}.xlsx"  # 修改为你的存储路径
        df.to_excel(excel_path, index=False)

        print(f"数据已成功保存至 {excel_path}")

    # Watch the performance
    # if __name__ == '__main__':
    #     env, _, _= make_aigc_env()
    #     policy.eval()
    #     collector = Collector(policy, env)
    #     result = collector.collect(n_episode=1)
    #     rews, lens = result["rews"], result["lens"]
    #     print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
