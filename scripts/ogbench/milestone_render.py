import os
import time
import subprocess
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from datetime import datetime



def setup_headless(display=':100', width=1024, height=768, depth=24) -> None:
    """
    Configure Xvfb for headless Mujoco rendering.
    Safe to call multiple times.
    """
    if 'DISPLAY' not in os.environ:
        subprocess.run(['pkill', 'Xvfb'], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.Popen(['Xvfb', display, '-screen', '0', f'{width}x{height}x{depth}', '-ac'])
        os.environ['DISPLAY'] = f'{display}.0'
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        # Optionally set a specific GPU; harmless if missing
        if 'EGL_DEVICE_ID' not in os.environ:
            os.environ['EGL_DEVICE_ID'] = '0'
        print(f'Xvfb {display} -screen 0 {width}x{height}x{depth} -ac')
        time.sleep(2)


@torch.no_grad()
def precompute_goal_embeddings(
    observations: np.ndarray,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    batch_size: int = 4096
) -> torch.Tensor:
    """
    Encode all observations into goal space using model.encode.

    Args:
        observations: np.ndarray of shape (N, obs_dim). Should match model's training normalization.
        model: DTAMP-like model exposing encode(obs)->(B, goal_dim)
        device: torch.device for computation
        batch_size: chunk size to bound memory

    Returns:
        Z: torch.Tensor of shape (N, goal_dim) on CPU (for distance search convenience)
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model_device = next(model.parameters()).device
    if model_device != device:
        # Prefer running on model's own device to avoid cross-device overhead
        device = model_device

    num_samples = observations.shape[0]
    goal_embeddings: List[torch.Tensor] = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        obs_chunk = torch.as_tensor(observations[start:end], dtype=torch.float32, device=device)
        g_chunk = model.encode(obs_chunk)
        goal_embeddings.append(g_chunk.detach().to('cpu'))
    Z = torch.cat(goal_embeddings, dim=0)
    return Z


@torch.no_grad()
def find_nearest_indices(
    milestones: np.ndarray,
    Z: torch.Tensor,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each milestone (goal vector), find the nearest dataset embedding index.

    Args:
        milestones: (M, goal_dim) np.ndarray or (M, goal_dim) torch-compatible
        Z: (N, goal_dim) torch.Tensor on CPU
        device: computation device for distance calc

    Returns:
        indices: (M,) np.ndarray of int indices into dataset
        distances: (M,) np.ndarray of float squared distances
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Z_device = Z.to(device)
    m = torch.as_tensor(milestones, dtype=torch.float32, device=device)
    # Use cdist for efficient batched nearest neighbor
    dists = torch.cdist(m, Z_device)
    nn_idx = torch.argmin(dists, dim=1)
    nn_dist = dists[torch.arange(dists.shape[0], device=device), nn_idx]
    return nn_idx.detach().cpu().numpy(), (nn_dist.detach().cpu().numpy() ** 2)


def _try_render_env_state(
    env,
    dataset: Dict[str, np.ndarray],
    index: int
) -> Optional[np.ndarray]:
    """
    Set the environment to the dataset state at `index` and render a frame.
    Requires `dataset` to include 'qpos', 'qvel', 'button_states' and env to support set_state.
    """
    qpos = dataset.get('qpos', None)
    qvel = dataset.get('qvel', None)
    btns = dataset.get('button_states', None)
    if qpos is None or qvel is None or btns is None:
        raise KeyError("raw_dataset must include 'qpos', 'qvel', and 'button_states' for state rendering")
    # Unwrap to reach base mujoco env
    inner_env = getattr(env, 'env', env)
    inner_env = getattr(inner_env, 'unwrapped', inner_env)
    if not hasattr(inner_env, 'set_state'):
        raise AttributeError('Environment does not support set_state for state rendering')
    inner_env.set_state(qpos[index], qvel[index], btns[index])
    frame = env.render()
    if not isinstance(frame, np.ndarray):
        raise RuntimeError('env.render() did not return an RGB array')
    return frame


def render_milestones_nearest(
    env,
    dataset: Dict[str, np.ndarray],
    model: torch.nn.Module,
    milestones: np.ndarray,
    out_dir: str,
    device: Optional[torch.device] = None,
    stride: int = 1,
    limit_dataset: Optional[int] = None,
    raw_dataset: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    Render images for planned milestones by selecting nearest dataset instances in goal space.

    Args:
        env: OGBench env (supports render). If it supports set_state and dataset contains states, exact frames are rendered.
        dataset: dict with at least 'observations'. Should match model's training-time normalization.
        model: DTAMP-like model exposing encode(obs)->(B, goal_dim)
        milestones: (M, goal_dim) planned goal vectors (np.ndarray or convertible)
        out_dir: directory to save images and index npz
        device: torch device for computation
        headless: whether to set up Xvfb for headless rendering
        stride: subsample dataset for index building (e.g., stride=5 uses every 5th sample)
        limit_dataset: cap the number of indexed samples (after stride)

    Returns:
        dict with keys: 'indices' (M,), 'distances' (M,), 'used_indices' (N_indexed,)
    """
    if raw_dataset is None:
        raise ValueError('raw_dataset (original train_dataset with qpos/qvel/button_states) is required for rendering')

    # Create timestamped subdirectory under out_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(out_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    observations = np.asarray(dataset['observations'], dtype=np.float32)
    if stride > 1:
        used_indices = np.arange(0, observations.shape[0], stride, dtype=np.int64)
        observations_index = observations[used_indices]
    else:
        used_indices = np.arange(observations.shape[0], dtype=np.int64)
        observations_index = observations
    if limit_dataset is not None:
        used_indices = used_indices[:limit_dataset]
        observations_index = observations_index[:limit_dataset]

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    Z = precompute_goal_embeddings(observations_index, model, device=device)
    if milestones is None or len(milestones) == 0:
        raise ValueError('milestones must be a non-empty array of shape (M, goal_dim)')
    if Z.shape[1] != np.asarray(milestones).shape[1]:
        raise ValueError(f'goal_dim mismatch: Z has {Z.shape[1]}, milestones has {np.asarray(milestones).shape[1]}')
    nn_local_idx, nn_dist = find_nearest_indices(milestones, Z, device=device)
    # Map local indices back to original dataset indices
    nn_global_idx = used_indices[nn_local_idx]

    # Try to render each selected index
    import imageio
    # Lazy import PIL for text overlay; fall back gracefully if unavailable
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        PIL_AVAILABLE = True
    except Exception:
        PIL_AVAILABLE = False

    def _overlay_milestone_number(image_np: np.ndarray, text: str) -> np.ndarray:
        if not PIL_AVAILABLE:
            return image_np
        try:
            img = Image.fromarray(image_np)
            draw = ImageDraw.Draw(img)
            try:
                # Common default font on many systems
                font = ImageFont.truetype("DejaVuSans.ttf", size=max(16, image_np.shape[0] // 32))
            except Exception:
                font = ImageFont.load_default()
            margin = max(8, image_np.shape[0] // 100)
            # Measure text size (use textbbox if available for better accuracy)
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = draw.textsize(text, font=font)
            x = max(0, image_np.shape[1] - text_w - margin)
            y = margin
            # Draw solid background for readability
            bg_padding = margin // 2
            bg_rect = [x - bg_padding, y - bg_padding, x + text_w + bg_padding, y + text_h + bg_padding]
            draw.rectangle(bg_rect, fill=(0, 0, 0))
            # Draw text in white
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            return np.asarray(img)
        except Exception:
            return image_np

    frames: List[np.ndarray] = []
    for i, ds_idx in enumerate(nn_global_idx.tolist()):
        frame = _try_render_env_state(env, raw_dataset, ds_idx)
        frame_annotated = _overlay_milestone_number(frame, f"Milestone {i}")
        imageio.imwrite(os.path.join(save_dir, f'milestone_{i:04d}.png'), frame_annotated)
        frames.append(frame_annotated)

    # Save indices and distances for analysis
    np.savez_compressed(
        os.path.join(save_dir, 'milestone_matches.npz'),
        indices=nn_global_idx,
        distances=nn_dist,
        used_indices=used_indices
    )

    # Create a video from frames: 2 seconds per milestone image
    # Use 30 fps and duplicate each frame 60 times for broad player compatibility
    video_path = os.path.join(save_dir, 'milestones.mp4')
    try:
        fps = 30
        duplicates_per_frame = 2 * fps  # 2 seconds per image
        with imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in frames:
                for _ in range(duplicates_per_frame):
                    writer.append_data(frame)
    except Exception:
        # Fallback: try writing a GIF if mp4 fails (e.g., missing ffmpeg)
        try:
            gif_path = os.path.join(save_dir, 'milestones.gif')
            # duration per frame in seconds
            imageio.mimsave(gif_path, frames, duration=2.0)
            video_path = gif_path
        except Exception:
            video_path = ''

    return {
        'indices': nn_global_idx,
        'distances': nn_dist,
        'used_indices': used_indices,
        'video_path': video_path,
        'output_dir': save_dir,
    }

