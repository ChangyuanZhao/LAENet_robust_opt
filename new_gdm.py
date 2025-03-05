# Import necessary libraries
import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env, make_mmwave_env, make_csi_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic, TransformerBlock, MoETransformerBlock
import warnings
import scipy.io as sio
import numpy as np
import pandas as pd

# Ignore warnings
warnings.filterwarnings('ignore')

# Define a function to get command line arguments
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1e6)#1e6
    parser.add_argument('-e', '--epoch', type=int, default=1e6)# 1000
    parser.add_argument('--step-per-epoch', type=int, default=1)# 100
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=16)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str, default='cuda:0')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=6)  # for diffusion chain 3 & 8 & 12
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])

    # With Expert: bc-coef True
    # Without Expert: bc-coef False
    # parser.add_argument('--bc-coef', default=False) # Apr-04-132705
    parser.add_argument('--bc-coef', default=False)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.4)#
    parser.add_argument('--prior-beta', type=float, default=0.4)#

    # Parse arguments and return them
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # create environments
    # env, train_envs, test_envs = make_mmwave_env(args.training_num, args.test_num)
    env, train_envs, test_envs = make_csi_env(args.training_num, args.test_num)
    args.state_shape = 96

    args.action_shape = 384
    args.max_action = 1.

    args.exploration_noise = args.exploration_noise * args.max_action
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # create actor
    # actor_net = MLP(
    #     state_dim=args.state_shape,
    #     action_dim=args.action_shape
    # )
    # actor_net = TransformerBlock(
    #     state_dim=args.state_shape,
    #     action_dim=args.action_shape
    # )
    actor_net = MoETransformerBlock(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )

    # Create critic
    # critic = DoubleCritic_attention(
    #     state_dim=args.state_shape,
    #     action_dim=args.action_shape
    # ).to(args.device)

    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)

    # Actor is a Diffusion model
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        bc_coef = args.bc_coef
    ).to(args.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd
    )



    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    ## Setup logging
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # def dist(*logits):
    #    return Independent(Normal(*logits), 1)

    # Define policy
    policy = DiffusionOPT(
        args.state_shape,
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        # dist,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        bc_coef=args.bc_coef,
        action_space=env.action_space,
        exploration_noise = args.exploration_noise,
    )

    # Load a previous policy if a path is provided
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # Setup buffer
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Trainer
    if not args.watch:
        # print("train begin")
        #
        # print("Collecting initial data...")
        # collected_data = train_collector.collect(n_step=args.step_per_collect)
        # #print(f"Collected data: {collected_data}")

        # print(args.epoch, args.step_per_epoch)

        args.step_per_collect = 1
        args.step_per_epoch = 1

        #print(args.step_per_collect)

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False
        )
        pprint.pprint(result)

        # 评估模型
        # final_performance = test_collector.collect(n_episode=args.test_num, render=False)

        name = "min"

        test_collector.reset_buffer()  # 清空历史数据

        result = test_collector.collect(n_episode=args.test_num, render=False)

        # 找到 avg_reward 最大的 buffer 索引
        best_idx = np.argmax([buf.info['avg_reward'] for buf in test_collector.buffer])

        # 获取对应的 f_forming 值
        f = test_collector.buffer[best_idx].info['f_forming']

        # 转换为 double complex 类型
        f = np.array(f, dtype=np.complex128)

        # 保存到 .mat 文件
        sio.savemat(f'./result/f_trained_proposed_{name}.mat', {'f': f})

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
        excel_path = f"./result/proposed_{name}.xlsx"  # 修改为你的存储路径
        df.to_excel(excel_path, index=False)

        print(f"数据已成功保存至 {excel_path}")

        # f = test_collector.buffer[0].info['f_forming']
        #
        # test_envs.get_env_attr("draw_beamforming")[0](f)


        # for i in range(len(test_collector.buffer)):
        #     print(f"Step {i}: {test_collector.buffer[i].info}")  # 逐个打印 buffer 里的 info

        # # 添加到返回结果
        # result["actions"] = actions
        #
        # print("Final Evaluation:", result)

        # print("Final Evaluation:", final_performance)

    # Watch the performance
    # python main.py --watch --resume-path log/default/diffusion/Jul10-142653/policy.pth
    # if __name__ == '__main__':
    #     policy.eval()
    #     collector = Collector(policy, env)
    #     result = collector.collect(n_episode=1) #, render=args.render
    #     print(result)
    #     rews, lens = result["rews"], result["lens"]
    #     print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    main(get_args())
