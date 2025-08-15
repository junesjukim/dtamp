import os
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
import wandb

import ogbench

from models.dtamp import DTAMP
from envs.ogbench_envs import OGBenchEnvWrapper


def evaluate(env, model, eval_episodes, threshold, time_limit=16, render=False, task_id=None):
    ep_returns = []
    pbar = tqdm(total=eval_episodes, desc='Evaluation')
    for _ in range(eval_episodes):
        options = {'render_goal': render}
        if task_id is not None:
            options['task_id'] = task_id
        obs, goal = env.reset(options=options, render_goal=render)
        if goal is None:
            goal = np.zeros_like(obs)
        done = False
        ep_return = 0.0
        milestones = model.planning(obs, goal, target_returns=None, num_samples=10)
        timestep = 0
        len_milestones = len(milestones)
        while not done:
            act, milestones = model.get_action(obs, goal, milestones, threshold=threshold)
            timestep += 1
            if len_milestones != len(milestones):
                len_milestones = len(milestones)
                timestep = 0
            if timestep > time_limit and len(milestones) > 1:
                milestones = model.planning(obs, goal, target_returns=None, num_samples=5)
                timestep = 0
            obs, rew, done, _ = env.step(act)
            if render:
                env.render()
            ep_return += rew
        ep_returns.append(ep_return)
        pbar.set_description(f'Evaluation - Avg return: {np.mean(ep_returns):.3f}')
        pbar.update(1)
    pbar.close()
    return np.mean(ep_returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scene-play-singletask-task2-v0')
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--override_config', type=str, default=None)
    parser.add_argument('--render', action='store_true', dest='render', default=False)
    parser.add_argument('--task_id', type=int, default=None)
    args = parser.parse_args()

    run_name = f"eval-{args.dataset}-{args.checkpoint_epoch or 'latest'}"
    wandb.init(project='dtamp-ogbench-eval', name=run_name, config=vars(args))

    checkpoint_dir = args.checkpoint_dir or os.path.join('checkpoints', f'dtamp_{args.dataset}_flowmatching')

    if args.config_path:
        config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    else:
        domain = args.dataset.split('-')[0]
        config = yaml.load(open(f'config/ogbench/{domain}.yml'), Loader=yaml.FullLoader)

    if args.override_config:
        key, value = args.override_config.split('=')
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass
        config[key] = value

    env, train_dataset, _ = ogbench.make_env_and_datasets(args.dataset)
    env = OGBenchEnvWrapper(env, train_dataset)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        sample_timesteps=config.get('sample_timesteps', config['diffuser_timesteps']),
        returns_condition=config['returns_condition'],
        condition_guidance_w=config['condition_guidance_w'],
        hidden_size=config['hidden_size'],
        model_type=config.get('model_type', 'diffusion'),
    ).to(device)

    if args.checkpoint_epoch:
        ckpt = torch.load(os.path.join(checkpoint_dir, f'checkpoint_{args.checkpoint_epoch}.pt'))
    else:
        ckpt = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
    state_dict = ckpt['model']
    for k in list(state_dict.keys()):
        if 'sample_' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    avg_return = evaluate(env, model, args.eval_episodes, config['threshold'], config['time_limit'], args.render, args.task_id)
    print(avg_return)
    wandb.log({'evaluation/avg_return': avg_return})
    wandb.finish()


if __name__ == '__main__':
    main()


