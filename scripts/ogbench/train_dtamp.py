import os
os.environ["WANDB_API_KEY"] = "b74af3d766f03aebd400095eec299dd945771d2b"


import os
import shutil
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb

import ogbench

from models.dtamp import DTAMP
from datasets.ogbench_dataset import OGBenchGCDataset
from envs.ogbench_envs import OGBenchEnvWrapper


def infer_domain_from_name(name):
    # e.g., 'scene-play-singletask-task2-v0' -> 'scene', 'cube-double-play-v0' -> 'cube'
    return name.split('-')[0]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='scene-play-singletask-task2-v0')
    parser.add_argument('--epochs_per_save', type=int, default=50)
    parser.add_argument('--exp-name', type=str, default=None)
    args = parser.parse_args()

    default_exp_name = f'dtamp_{args.dataset}_flowmatching'
    exp_name = args.exp_name if args.exp_name else default_exp_name
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join('logs', exp_name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    domain = infer_domain_from_name(args.dataset)
    config = yaml.load(open(f'config/ogbench/{domain}.yml'), Loader=yaml.FullLoader)

    wandb.init(project='dtamp-ogbench', name=exp_name, config={**vars(args), **config})

    # OGBench API: env + datasets (train/val)
    env, train_dataset, _ = ogbench.make_env_and_datasets(args.dataset)
    env = OGBenchEnvWrapper(env, train_dataset)

    dataset = OGBenchGCDataset(env.get_dataset(), config['max_interval'], config['horizon'])
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)

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
        returns_condition=config['returns_condition'],
        condition_guidance_w=config['condition_guidance_w'],
        hidden_size=config['hidden_size'],
        model_type=config.get('model_type', 'diffusion'),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    model.train()
    n_updates, epoch = 0, 0
    total_updates = config['updates_per_epoch'] * config['epochs']
    pbar = tqdm(total=config['updates_per_epoch'], desc=f'Epoch {epoch}')
    while n_updates < total_updates:
        for batch in data_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            loss, logs = model.loss(batch, warmup=n_updates < config['warmup_updates'])

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tag = 'train' if n_updates < config['warmup_updates'] else 'finetune'
            wandb.log({f'{tag}/{k}': v for k, v in logs.items()}, step=n_updates)

            pbar.update(1)
            n_updates += 1

            if n_updates % config['updates_per_epoch'] == 0:
                pbar.close()
                epoch += 1
                model.eval()

                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'n_updates': n_updates,
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint.pt'))
                if epoch % args.epochs_per_save == 0:
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt'))

                if n_updates == total_updates:
                    break
                model.train()
                pbar = tqdm(total=config['updates_per_epoch'], desc=f'Epoch {epoch}')


if __name__ == '__main__':
    # WANDB_API_KEY should be set via environment variable if needed
    train()


