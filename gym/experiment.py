import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # print(trajectories)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    # f = open('log.txt', 'w')
    cnt = 0
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        # print(len(path['observations']))
        # print()
        # cnt += len(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    # traj_lens一条轨迹的长度，returns记录一条轨迹的回报总和
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    # print(len(states))
    # print(cnt)
    states = np.concatenate(states, axis=0)
    # print(len(states))
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # num_timesteps 是所有traj的states数量的和，也即states总数量
    num_timesteps = sum(traj_lens)
    # print(num_timesteps)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # K是sequence中取的state序列的长度
    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    # pct_traj = variant.get('pct_traj', 1.)
    pct_traj = 0.01

    # only train on top pct_traj trajectories (for %BC experiment)
    # print(num_timesteps)
    # 前pct_traj%长的traj被用来训练%BC，共num_timesteps个
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    # print(num_timesteps)
    # return从低到高的returns序列排序的序列 对应的原序列的 索引id
    sorted_inds = np.argsort(returns)  # lowest to highest
    # (%BC)使用的traj的数量
    num_trajectories = 1
    # sorted_inds[-1]：最大的return对应的原traj的index
    # timesteps初始为return最大的traj的长度
    # 经过处理后，timesteps为所有使用的traj的长度和
    timesteps = traj_lens[sorted_inds[-1]]
    #
    ind = len(trajectories) - 2
    # sorted_inds[len(trajectories) - 2]： return第二大的traj的原序列index
    # sorted_inds[ind]：
    # ind >= 0 ：限制最多全用
    # timesteps + traj_lens[sorted_inds[ind]]： traj按return排过之后越大就越先加起来的len和
    # 只用最多 num_timesteps 个 states
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        # 其实这里用了<会使最后一个无法被收入
        # print(num_timesteps,traj_lens[sorted_inds[ind]])
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1

    # 把ind的list的后num_trajectories提取出来，长度：num_trajectories
    sorted_inds = sorted_inds[-num_trajectories:]
    # print(f'num_trajectories:{num_trajectories}')

    # print(sorted_inds)
    # print(traj_lens[sorted_inds])
    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    # print(p_sample)
    # print(sum(p_sample))

    def get_batch(batch_size=256, max_len=K):
        # numpy.random.choice(a, size=None, replace=True, p=None)
        # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
        # replace:True表示可以取相同数字，False表示不可以取相同数字
        # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
        batch_inds = np.random.choice(
            # num_trajectories个样本的index的list
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            # 长度：num_trajectories
            p=p_sample,  # reweights so we sample according to timesteps
        )
        # print(f"batch_size:{batch_size}")
        # print(f'num_trajectories:{num_trajectories}')
        # print(f'len(batch_inds):{len(batch_inds)}')

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        breakk = 0
        cnt_out = 0
        cnt_in = 0

        for i in range(batch_size):
            # print(len(trajectories))
            # batch_inds[i] ： 第i个batch的sample出的某一个在num_trajectories个traj中的traj的随机index
            # sorted_inds[batch_inds[i]] ： 最长的num_trajectories个traj的原序列index的list中的第batch_inds[i]个
            # trajectories[int(sorted_inds[batch_inds[i]])] ： 第batch_inds[i]长的traj
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            # si为随机出来的某一条traj的在traj序列里的index
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # print(traj['rewards'].shape)              #:(len,)
            # print(traj['observations'].shape)         #:(len, state_dim)
            # print(traj['next_observations'].shape)    #:(len, state_dim)
            # print(traj['actions'].shape)              #:(len, action_dim)
            # print(traj['terminals'].shape)            #:(len,)

            # get sequences from dataset

            # state_dim = env.observation_space.shape[0]
            # print(traj['observations'][si:si + max_len])
            # reshape: 1, K, env.observation_space.shape[0]
            # 1, max_len, state_dim
            # array([
            #   [
            #       [x0101,x0102,x0103,x0104,x0105,x0106,x0107,x0108,x0109,x0110,x0111],
            #       [x0201,x0202,x0203,x0204,x0205,x0206,x0207,x0208,x0209,x0210,x0211],
            #       ...
            #       ...
            #       ...(max-len=20 时)
            #       [x2001,x2002,x2003,x2004,x2005,x2006,x2007,x2008,x2009,x2010,x2011]
            #   ]
            # ])
            # 作为一个array，最外面的一层[]是无实义的，换句话说，每一个[]内部的才是实际计算的维度
            # np.array([1]).shape ： (1,)
            # np.array([[1,2]]).shape ： (1, 2)
            # np.array([[[1,2,3],[1,2,3]]]).shape ： (1, 2, 3)
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            # print(s)
            # if breakk:
            #     break
            # breakk = True

            # print(s)
            # break

            # act_dim = env.action_space.shape[0]
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            # rewards维度为1
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                # print(d)
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            # timesteps 补充最新的若干（上述的reshape中的-1那个维的维数）个states的序列的在原来全traj全states的全收录list里的的编号，shape[-1] 一般为 K， 不够长了则看情况，剩多少states就是多少
            #
            # if s[-1].shape[1] != max_len:
            #     print(len(s))
            #     print(si)
            #     print(traj['rewards'].shape[0])
            #     print(si + max_len)
            #     print(traj['rewards'][si:si + max_len])
            #     print(traj['rewards'][si:si + 100000000000000]) # 和上面其实一样
            #     print(traj['rewards'][si + max_len]) # 会抛异常
            #     print(f"s[-1].shape:{s[-1].shape}")
            #     print(f"s[-1].shape[1]:{s[-1].shape[1]}")
            #     break
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))

            # print(timesteps)
            # if breakk == 1:
            #     break
            # breakk += 1

            # max_ep_len = 10
            # print(max_ep_len)
            # print(timesteps[-1])
            # timesteps[-1]这个list里面所有不小于max_ep_len的全部变成max_ep_len-1
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            # print(timesteps[-1])
            # res[t] = discount_cumsum(x, gamma):
            # res[t] = x[t + 0] + gamma(x[t + 1] + gamma * (x[t + 2] + gamma *(x[t + 3] + gammma * ( ... )))))
            # 从si开始的rewards衰减和的序列
            # 上述衰减和序列
            # rtg的append的是最新states截取序列的长度+1，s[-1].shape[1]看情况，一般为max_len，否则为剩余长度
            # :s[-1].shape[1] + 1 --- 长度通常为max_len+1，reshape过后通常为.reshape(1, max_len+1, 1)
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            # cnt_out += 1
            # print(rtg[-1].shape)：通常为21，少数为20及以下
            # print(s[-1].shape)：通常为20，少数为20以下
            # 上面两个20以max_len为准
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                # 这个if实际上找的是等于
                # cnt_in += 1
                # print(cnt_in, cnt_out, batch_size)

                # print(rtg[-1].shape)
                # print(s[-1].shape)
                # 上述两个出现在if里都是不超过max_len且相等

                # (1, 1, 1)是shape，相当于在rtg后面补0
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
                # print(rtg[-1].shape)
            # 实际上保证rtg[-1].shape[-1] = s[-1].shape[-1]

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # states 前面补0.(然后再标准化)
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            # action前面补-10.
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            # if max_len != tlen:
            #     print(a[-1])
            # rwards前面补0.
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            # d前面补2.
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            # rtg前面补0.，然后再scale标准化？
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            # timesteps前面补0
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            # mask前面补0.，有的全都对应设为1.
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        # print(s[0].shape) : (1, max_len, state_dim)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        # print(s.shape) : (batch_size, max_len, state_dim)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='bc')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
