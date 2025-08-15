import numpy as np
from gym.spaces import Box


class OGBenchEnvWrapper:
    """
    Environment wrapper for OGBench (Gymnasium API).
    - Normalizes observations using dataset min/max to match training distribution.
    - Bridges Gymnasium step() (terminated, truncated) to Gym-style done.
    - Returns a goal observation from env.reset() info when available (GC tasks).
    """

    def __init__(self, env, dataset_dict, normalize_obs=True):
        self.env = env
        self.dataset = dict(dataset_dict)
        self.normalize_obs = normalize_obs

        observations = self.dataset['observations']
        self.max_obs = np.max(observations, axis=0)
        self.min_obs = np.min(observations, axis=0)
        self._denom = (self.max_obs - self.min_obs)
        self._denom[self._denom == 0.0] = 1.0

        if normalize_obs:
            max_obs = self.max_obs.reshape(1, -1)
            min_obs = self.min_obs.reshape(1, -1)
            denom = (max_obs - min_obs)
            denom[denom == 0.0] = 1.0
            self.dataset['observations'] = 2.0 * (observations - min_obs) / denom - 1.0

    def get_dataset(self):
        return self.dataset

    def _normalize(self, obs):
        return 2.0 * (obs - self.min_obs) / self._denom - 1.0

    def reset(self, options=None, render_goal=False):
        options = options or {}
        # Ensure render_goal flag is propagated when desired
        if 'render_goal' not in options:
            options['render_goal'] = render_goal
        ob, info = self.env.reset(options=options)
        goal = info.get('goal', None)
        if self.normalize_obs:
            ob = self._normalize(ob)
            if goal is not None:
                try:
                    # Normalize goal only when shape matches observation
                    if np.shape(goal) == np.shape(ob):
                        goal = self._normalize(goal)
                except Exception:
                    pass
        return ob, goal

    def step(self, act):
        ob, rew, terminated, truncated, info = self.env.step(act)
        done = terminated or truncated
        if self.normalize_obs:
            ob = self._normalize(ob)
        return ob, rew, done, info

    @property
    def observation_space(self):
        dim = self.dataset['observations'].shape[1]
        return Box(-np.ones(dim), np.ones(dim), shape=(dim,))

    @property
    def action_space(self):
        return self.env.action_space

    def render(self, mode='human'):
        return self.env.render()


