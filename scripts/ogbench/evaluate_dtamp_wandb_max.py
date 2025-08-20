import os
os.environ["WANDB_API_KEY"] = "b74af3d766f03aebd400095eec299dd945771d2b"

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
    successes = []
    pbar = tqdm(total=eval_episodes, desc='Evaluation')

    for episode_idx in range(eval_episodes):
        # Reset with optional task and goal rendering
        options = {'render_goal': render}
        if task_id is not None:
            options['task_id'] = task_id
        obs, goal = env.reset(options=options, render_goal=render)
        if goal is None:
            goal = np.zeros_like(obs)

        # Episode trackers
        done = False
        ep_return = 0.0
        rewards = []
        step_in_episode = 0
        time_limit_exceed_steps = []  # step index (1-based) when time_limit condition triggered
        milestones_len_after_exceeds = []  # len(milestones) immediately after applying the condition
        last_info = {}

        # Initial planning
        milestones = model.planning(obs, goal, target_returns=None, num_samples=10)
        timestep = 0
        len_milestones = len(milestones)
        initial_milestones_len = len_milestones

        while not done:
            act, milestones = model.get_action(obs, goal, milestones, threshold=threshold)

            # Safety: clip to action space bounds
            act = np.clip(act, env.action_space.low, env.action_space.high)

            # Progress local counters
            step_in_episode += 1
            timestep += 1

            # If milestones list length changed, reset local time counter
            if len_milestones != len(milestones):
                len_milestones = len(milestones)
                timestep = 0

            # Handle time limit condition and record detailed telemetry
            if timestep > time_limit and len(milestones) > 1:
                # In OGBench eval, we drop the first milestone instead of replanning
                milestones = milestones[1:]
                # Record when this happened and how many milestones remain
                time_limit_exceed_steps.append(int(step_in_episode))
                milestones_len_after_exceeds.append(int(len(milestones)))
                # Reset local timer after adjustment
                timestep = 0

            # Environment step
            obs, rew, done, info = env.step(act)
            last_info = info
            if render:
                env.render()
            ep_return += float(rew)
            rewards.append(float(rew))

        # Episode summary
        ep_success = float(last_info.get('success', 0.0))
        ep_returns.append(ep_return)
        successes.append(ep_success)

        # Build rich episode logs
        rewards_table = wandb.Table(
            columns=["step", "reward"],
            data=[[step, rewards[step - 1]] for step in range(1, step_in_episode + 1)]
        )
        exceed_events_table = wandb.Table(
            columns=["event_index", "step", "milestones_len_after"],
            data=[[i + 1, s, m] for i, (s, m) in enumerate(zip(time_limit_exceed_steps, milestones_len_after_exceeds))]
        )

        # Episode-specific plots (each episode is a separate series/panel)
        steps_list = list(range(1, step_in_episode + 1))
        rewards_series_plot = wandb.plot.line_series(
            xs=[steps_list],
            ys=[rewards],
            keys=[f"ep-{episode_idx}"],
            title=f"Episode {episode_idx} Rewards",
            xname="step",
        )

        exceed_events_scatter_table = wandb.Table(
            columns=["step", "milestones_len_after"],
            data=[[s, m] for s, m in zip(time_limit_exceed_steps, milestones_len_after_exceeds)]
        )
        exceed_events_scatter_plot = wandb.plot.scatter(
            exceed_events_scatter_table,
            x="step",
            y="milestones_len_after",
            title=f"Episode {episode_idx} Time-limit Events",
        )

        episode_log = {
            "episode/index": int(episode_idx),
            "episode/ep_return": float(ep_return),
            "episode/success": ep_success,
            "episode/total_steps": int(step_in_episode),
            "episode/initial_milestones_len": int(initial_milestones_len),
            # time_limit telemetry
            "episode/time_limit_exceeded_count": int(len(time_limit_exceed_steps)),
            "episode/time_limit_exceeded_steps": time_limit_exceed_steps,
            "episode/milestones_len_after_exceeds": milestones_len_after_exceeds,
            # detailed sequences
            "episode/rewards": rewards,
            "episode/rewards_table": rewards_table,
            "episode/time_limit_events_table": exceed_events_table,
            # per-episode plot panels (distinct per episode)
            f"episode/plot/rewards_ep_{episode_idx}": rewards_series_plot,
            f"episode/plot/time_limit_events_ep_{episode_idx}": exceed_events_scatter_plot,
        }

        wandb.log(episode_log)

        # Progress bar
        pbar.set_description(f'Evaluation - Avg return: {np.mean(ep_returns):.3f}')
        pbar.update(1)

    pbar.close()
    avg_return = float(np.mean(ep_returns))
    success_rate = float(np.mean(successes) * 100.0)
    return avg_return, success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scene-play-singletask-task2-v0')
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--override_config', type=str, default=None)
    parser.add_argument('--render', action='store_true', dest='render', default=False)
    parser.add_argument('--task_id', type=int, default=None)
    args = parser.parse_args()

    run_name = f"eval-maxlog-{args.dataset}-{args.checkpoint_epoch or 'latest'}"
    wandb.init(project='dtamp-ogbench-eval_wandb_max', name=run_name, config=vars(args))

    checkpoint_dir = args.checkpoint_dir or os.path.join('checkpoints', f'dtamp_{args.dataset}_flowmatching')

    # Load config
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

    # Env
    env, train_dataset, _ = ogbench.make_env_and_datasets(args.dataset)
    env = OGBenchEnvWrapper(env, train_dataset)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
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

    # Load checkpoint
    if args.checkpoint_epoch:
        ckpt = torch.load(os.path.join(checkpoint_dir, f'checkpoint_{args.checkpoint_epoch}.pt'))
    else:
        ckpt = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
    state_dict = ckpt['model']
    # Remove diffuser sample params if present
    for k in list(state_dict.keys()):
        if 'sample_' in k:
            del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Evaluate
    avg_return, success_rate = evaluate(
        env,
        model,
        args.eval_episodes,
        config['threshold'],
        config['time_limit'],
        args.render,
        args.task_id,
    )

    # Final logs
    wandb.log({'evaluation/avg_return': avg_return, 'evaluation/success_rate': success_rate})
    print({'avg_return': avg_return, 'success_rate': success_rate})
    wandb.finish()


if __name__ == '__main__':
    main()


