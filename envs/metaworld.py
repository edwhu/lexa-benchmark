
import numpy as np
import gym
import random
import itertools
from itertools import combinations
from envs.base_envs import BenchEnv
from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer
from envs.utils import parse, translate
import re

class MetaWorld(BenchEnv):
    def __init__(self, name, action_repeat=2, use_goal_idx=False, log_per_goal=False):
        super().__init__(action_repeat)
        from mujoco_py import MjRenderContext
        # import metaworld.envs.mujoco.sawyer_xyz as sawyer
        import metaworld.envs.mujoco.sawyer_xyz.v1 as sawyer
        import metaworld.envs.mujoco.sawyer_xyz.v2 as sawyerv2
        domain, task = name.split('_', 1)

        self._task_keys, task = parse(task, ['closeup', 'frontview2', 'frontview3','imed', 'ieasy', 'boxedtop', 'boxedfront', 'boxedside', 'sideviewblock', 'topdown', 'nogoallist', 'simplerew', 'fullstaterew'])
        self.task_type = self.get_task_type(task)

        self.v2 = v2 = 'V2' in task
        kwargs = dict()
        if self._task_keys['fullstaterew']:
          kwargs['full_state_reward'] = True
        
        with self.LOCK:
          task = translate(task, {'pickbin': 'SawyerBinPickingEnv'})
          if not v2:

            if self.task_type in ['reach', 'push']:
                self._env = sawyer.SawyerReachPushPickPlaceEnv(task_type = self.task_type, **kwargs)
            else:
                self._env = getattr(sawyer, task)(**kwargs)
          else:
            self._env = getattr(sawyerv2, task)(**kwargs)
        self._env.random_init = False

        if self._task_keys['boxedfront']:
          self._env.mocap_low = (-0.5, 0.40, 0.07)
          self._env.mocap_high = (0.5, 0.8, 0.5)

        elif self._task_keys['boxedtop']:
          self._env.mocap_low = (-0.5, 0.40, 0.07)
          self._env.mocap_high = (0.5, 0.8, 0.3)

        elif self._task_keys['boxedside']:
          self._env.mocap_low = (-0.2, 0.4, 0.07)
          self._env.mocap_high = (0.8, 0.8, 0.5)

        self._env.goals = get_sawyer_reach_goals()

        self._action_repeat = action_repeat
        self._width = 64
        self._size = (self._width, self._width)

        #self._offscreen = MjRenderContext(self._env.sim, True, get_device_id(), RENDERER)
        self.renderer = DMRenderer(self._env.sim, camera_settings=dict(
              distance=0.6, lookat=[0, 0.65, 0], azimuth=90, elevation=41+180))
        
        if self._task_keys['frontview3']:
          self.renderer._camera_settings = dict(
              distance=0.6, lookat=[0, 0.65, 0], azimuth=90, elevation=41+180)

        self.use_goal_idx = use_goal_idx
        self.log_per_goal = log_per_goal
        self.rendered_goal = False

    def get_task_type(self, task):
      if 'Push' in task:
        return 'push'
      if 'Reach' in task:
        return 'reach'
      if 'TwoBlockBin' in task:
        return 'twoblockbin'
      if 'bin' in task.lower():
        return 'pickbin'
      if 'BlockPicking' in task:
        return 'pickblock'
      if 'Drawer' in task:
        return 'drawer'

    def step(self, action):
        total_reward = 0.0
        for step in range(self._action_repeat):
            state, reward, done, info = self._env.step(action)
            total_reward += min(reward, 100000)
            if done:
                break
        obs = self._get_obs(state)
        for k, v in obs.items():
          if 'metric_' in k:
            info[k] = v
        return obs, total_reward, done, info

    def reset(self):
      if self._task_keys['imed']:
        self._env.init_config['hand_init_pos'] = self._env.hand_init_pos = np.array([0, .6, .1])
      elif self._task_keys['ieasy']:
        self._env.init_config['hand_init_pos'] = self._env.hand_init_pos = np.array([0., .6, .05])
      self.rendered_goal = False
      if not self.use_goal_idx:
        self._env.goal_idx = np.random.randint(len(self._env.goals))
      return super().reset()

    def _get_obs(self, state):
      obs = super()._get_obs(state)
      obs['image_goal'] = self.render_goal()
      obs['goal'] = self._env.goals[self._env.goal_idx]
      if self.log_per_goal:
        obs = self._env.add_pertask_success(obs)
      elif self.use_goal_idx:
        obs = self._env.add_pertask_success(obs, self._env.goal_idx)
      return obs

    def render_goal(self):
        if self.rendered_goal:
            return self.rendered_goal_obj
        # TODO use self.render_state

        obj_init_pos_temp = self._env.init_config['obj_init_pos'].copy()
        goal = self._env.goals[self._env.goal_idx]

        self._env.init_config['obj_init_pos'] = goal[3:]
        self._env.obj_init_pos = goal[3:]
        self._env.hand_init_pos = goal[:3]
        self._env.reset_model()
        action = np.zeros(self._env.action_space.low.shape)
        if self.task_type in ['pickblock']:
            action[-1] = 1
            for _ in range(7):
                state, reward, done, info = self._env.step(action)
        else:
            state, reward, done, info = self._env.step(action)

        goal_obs = self.render_offscreen()
        self._env.hand_init_pos = self._env.init_config['hand_init_pos']
        self._env.init_config['obj_init_pos'] = obj_init_pos_temp
        self._env.obj_init_pos = self._env.init_config['obj_init_pos']
        self._env.reset()

        self.rendered_goal = True
        self.rendered_goal_obj = goal_obs
        return goal_obs

    def render_state(self, state):
      assert (len(state.shape) == 1)
      # Save init configs
      hand_init_pos = self._env.hand_init_pos
      obj_init_pos = self._env.init_config['obj_init_pos']
      # Render state
      hand_pos, obj_pos, hand_to_goal = np.split(state, 3)
      self._env.hand_init_pos = hand_pos
      self._env.init_config['obj_init_pos'] = obj_pos
      self._env.reset_model()
      obs = self._get_obs(state)
      # Revert environment
      self._env.hand_init_pos = hand_init_pos
      self._env.init_config['obj_init_pos'] = obj_init_pos
      self._env.reset()
      return obs['image']

    def render_states(self, states):
      assert (len(states.shape) == 2)
      imgs = []
      for s in states:
        img = self.render_state(s)
        imgs.append(img)
      return np.array(imgs)

    def set_goal_idx(self, idx):
      self._env.goal_idx = idx

    def get_goal_idx(self):
      return self._env.goal_idx

    def get_goals(self):
      return self._env.goals

def get_sawyer_reach_goals():
  goal_grid_size = 4
  obj_pos = [0, 0.6, 0.02]
  x_lims = [-0.2, 0.2] ; y_lims = [0.4, 0.8] ; z = 0.0
  goals = []
  for x in np.linspace(x_lims[0], x_lims[1], goal_grid_size):
    for y in np.linspace(y_lims[0], y_lims[1], goal_grid_size):
      hand_pos = np.array([x,y,z])
      goals.append(np.concatenate([hand_pos, obj_pos]))
  return goals

