import numpy as np
import torch
import wandb

from models.dtamp import DTAMP

import gym
import d4rl

import yaml
from tqdm import tqdm


def evaluate(env, model, eval_episodes, threshold, time_limit=16, render=False):
    ep_returns = []
    pbar = tqdm(total=eval_episodes, desc=f'Evaluation')
    for _ in range(eval_episodes):
        obs, goal = env.reset()
        done = False
        ep_return = 0
        milestones = model.planning(obs, goal, target_returns=None)
        timestep = 0
        len_milestones = len(milestones)
        while not done:
            act, milestones = model.get_action(obs, milestones, threshold=threshold)
            timestep += 1
            if len_milestones != len(milestones):
                len_milestones = len(milestones)
                timestep = 0
            if timestep > time_limit and len(milestones) > 1:
                milestones = milestones[1:]
                timestep = 0
            obs, rew, done, _ = env.step(act)
            #print("time :", timestep, " | rew :", rew)
            if render:
                env.render(mode='human')
            ep_return += rew
        ep_returns.append(ep_return)
        pbar.set_description(f'Evaluation - Avg return: {np.mean(ep_returns):.3f}')
        pbar.update(1)
    pbar.close()
    return np.mean(ep_returns)


if __name__ == '__main__':
    import argparse
    import os

    os.environ["WANDB_API_KEY"] = "b74af3d766f03aebd400095eec299dd945771d2b"

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--env', type=str, default='antmaze-medium-play-v2')
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--override_config', type=str, default=None)
    parser.add_argument('--render', action='store_true', dest='render', default=False)
    args = parser.parse_args()

    wandb.init(
        project="dtamp-evaluation",
        name=f"eval-{args.env}-{args.checkpoint_epoch or 'latest'}",
        config=vars(args)
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % args.gpu

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = os.path.join('checkpoints', f'dtamp_{args.env}')

    if args.config_path:
        config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    else:
        domain = args.env.split('-')[0]
        config = yaml.load(open(f'config/d4rl/{domain}.yml'), Loader=yaml.FullLoader)
    
    if args.override_config:
        key, value = args.override_config.split('=')
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        config[key] = value

    env = gym.make(args.env)
    domain = args.env.split('-')[0]
    if domain == 'antmaze':
        from envs.d4rl_envs import AntmazeEnvWrapper as EnvWrapper
    elif domain == 'kitchen':
        from envs.d4rl_envs import KitchenEnvWrapper as EnvWrapper
    else:
        raise NotImplementedError

    env = EnvWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda')

    model = DTAMP(
        state_dim=state_dim,
        act_dim=action_dim,
        goal_dim=config['goal_dim'],
        visual_perception=False,
        horizon=config['horizon'],
        n_critics=config['n_critics'],
        rl_coeff=config['rl_coeff'],
        kl_coeff=config['kl_coeff'],
        decoder_coeff=config['decoder_coeff'],
        diffuser_coeff=config['diffuser_coeff'],
        predict_epsilon=config['predict_epsilon'],
        diffuser_timesteps=config['diffuser_timesteps'],
        sample_timesteps=config.get('sample_timesteps', config['diffuser_timesteps']), # Add this line
        returns_condition=config['returns_condition'],
        condition_guidance_w=config['condition_guidance_w'],
        hidden_size=config['hidden_size'],
        model_type=config.get('model_type', 'diffusion')
    ).to(device)
    if args.checkpoint_epoch:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint_%d.pt' % args.checkpoint_epoch))
    else:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
    state_dict = checkpoint['model']
    # remove diffuser sample parameter names from state_dict
    for k in list(state_dict.keys()):
        if 'sample_' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    avg_return = evaluate(env, model, args.eval_episodes,
                          config['threshold'], config['time_limit'], args.render)
    print(avg_return)
    wandb.log({'evaluation/avg_return': avg_return})
    wandb.finish()
