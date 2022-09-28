
# Requires installing OpenAI gym and safety gym.
import os
import copy
import pickle
import numpy as np

import safety_gym
import gym
from gym import register
from gym import Wrapper
import torch

from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging import logger

# To use: return a dict of keys and default values which sometimes appear in
# the wrapped env's env_info, so this env always presents those values (i.e.
# make keys and values keep the same structure and shape at all time steps.)
# Here, a dict of kwargs to be fed to `sometimes_info` should be passed as an
# env_kwarg into the `make` function, which should be used as the EnvCls.
def sometimes_info(*args, **kwargs):
    # e.g. Feed the env_id.
    # Return a dictionary (possibly nested) of keys: default_values
    # for this env.
    return dict(cost_exception=0, goal_met=False)


def print_dict(d:dict, prefix=''):
    print(prefix + '{')
    for k, v in d.items():
        if isinstance(v, dict):
            new_prefix = prefix + '\t'
            print(f'{prefix}{k}: ')
            print_dict(v, prefix=new_prefix)
        else:
            print(f'{prefix}{k}: {v}')
    print(prefix + '}')


class SafetyGymEnvWrapper(Wrapper):

    """
    Hack:
        - Robot Position: env.world.robot_pos()
        - Goal Position: env.goal_pos
        -

    Functions:
        - env.obs_compass(pos): compass observation
        - env.sim.get_state(): get full state
        - env.sim.set_state(value): set full state

    """

    def __init__(self, env, sometimes_info_kwargs, obs_prev_cost, cost_limit=25.0, discount=0.99):
        super().__init__(env)
        self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._obs_prev_cost = obs_prev_cost
        self._prev_cost = 0.  # Concat this into the observation.
        self._cost_limit = cost_limit
        self._discount = discount
        self._remain_cost = self._cost_limit
        self._remain_discounted_cost = self._cost_limit
        self._n_steps = 0

        # trace_dir = os.path.join(logger.get_snapshot_dir(), 'TrajTrace')
        # if not os.path.isdir(trace_dir):
        #     os.makedirs(trace_dir)
        # self._traj_tracer = SafetyGymTrajTracer(trace_dir)

        obs = env.reset()
        # self._traj_tracer.reset(env=self.env)

        # Some edited version of safexp envs defines observation space only
        # after reset, so expose it here (what base Wrapper does):
        self.observation_space = env.observation_space
        if isinstance(obs, dict):  # and "vision" in obs:
            self._prop_keys = [k for k in obs.keys() if k != "vision"]
            obs = self.observation(obs)
            prop_shape = obs["prop"].shape
            # if obs_prev_cost:
            #     assert len(prop_shape) == 1
            #     prop_shape = (prop_shape[0] + 1,)
            obs_space = dict(
                prop=gym.spaces.Box(-1e6, 1e6, prop_shape,
                    obs["prop"].dtype))
            if "vision" in obs:
                obs_space["vision"] = gym.spaces.Box(0, 1, obs["vision"].shape,
                    obs["vision"].dtype)
            # GymWrapper will in turn convert this to rlpyt.spaces.Composite.
            self.observation_space = gym.spaces.Dict(obs_space)
        elif obs_prev_cost:
            if isinstance(obs, dict):
                self.observation_space.spaces["prev_cost"] = gym.spaces.Box(
                    -1e6, 1e6, (1,), np.float32)
            else:
                obs_shape = obs.shape
                assert len(obs_shape) == 1
                obs_shape = (obs_shape[0] + 1,)
                self.observation_space = gym.spaces.Box(-1e6, 1e6, obs_shape,
                    obs.dtype)
        self._cum_cost = 0.

    def step(self, action):
        o, r, d, info = self.env.step(action)
        o = self.observation(o)  # Uses self._prev_cost
        self._prev_cost = info.get("cost", 0)

        # self._traj_tracer.step(
        #     env=self.env,
        #     reward=r,
        #     cost=self._prev_cost,
        #     done=d,
        # )

        self._cum_cost += self._prev_cost
        self._remain_cost = max(self._remain_cost - self._prev_cost, 1.0)
        self._remain_discounted_cost = min(max((self._remain_discounted_cost - self._prev_cost) / self._discount, 1.0), self._cost_limit)
        info["cum_cost"] = self._cum_cost
        info["remain_cost"] = self._remain_cost
        info["remain_discounted_cost"] = self._remain_discounted_cost
        # Try to make info dict same key structure at every step.
        info = infill_info(info, self._sometimes_info)
        for k, v in info.items():
            if isinstance(v, float):
                info[k] = np.dtype("float32").type(v)  # In case computing on.
        # Look inside safexp physics env for this logic on env horizon:
        info["timeout"] = d and (self.env.steps >= self.env.num_steps)
        # info["timeout_next"] = not d and (
        #     self.env.steps == self.env.num_steps - 1)
        return o, r, d, info

    def reset(self):
        self._prev_cost = 0.
        self._cum_cost = 0.
        self._remain_cost = self._cost_limit
        self._remain_discounted_cost = self._cost_limit
        self._n_steps = 0
        obs = self.observation(self.env.reset())
        # self._traj_tracer.reset(env=self.env)
        return obs

    def observation(self, obs):
        if isinstance(obs, dict):  # and "vision" in obs:
            # flatten everything else than vision.
            obs_ = dict(
                prop=np.concatenate([obs[k].reshape(-1)
                    for k in self._prop_keys])
            )
            if "vision" in obs:
                # [H,W,C] --> [C,H,W]
                obs_["vision"] = np.transpose(obs["vision"], (2, 0, 1))
            if self._obs_prev_cost:
                obs_["prop"] = np.append(obs_["prop"], self._prev_cost)
            obs = obs_
        elif self._obs_prev_cost:
            obs = np.append(obs, self._prev_cost)
        return obs

    def get_full_state(self):
        world_config_dict = copy.deepcopy(self.env.world_config_dict)
        joint_state = self.env.sim.get_state() #internal state of the robot and objects

        # pos_state =
        # vel_state =
        # mat_state =








def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info


def register_more_envs(id_config: list):
    for d in id_config:
        register(id=d['id'], entry_point='safety_gym.envs.mujoco:Engine', kwargs={'config': d['config']})


def safety_gym_make(*args, sometimes_info_kwargs=None, obs_prev_cost=True,
        obs_version="default", **kwargs):
    assert obs_version in ["default", "vision", "vision_only", "no_lidar",
        "no_constraints"]
    if obs_version != "default":
        eid = kwargs["id"]  # Must provide as kwarg, not arg.
        names = dict(  # Map to my modification in safety-gym suite.
            vision="Vision",
            vision_only="Visonly",
            no_lidar="NoLidar",
            no_constraints="NoConstr",
        )
        name = names[obs_version]
        # e.g. Safexp-PointGoal1-v0 --> Safexp-PointGoal1Vision-v0
        kwargs["id"] = eid[:-3] + name + eid[-3:]
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')
    return GymEnvWrapper(SafetyGymEnvWrapper(
        gym.make(*args, **kwargs),
        sometimes_info_kwargs=sometimes_info_kwargs or dict(),
        obs_prev_cost=obs_prev_cost),
    )


class SafetyGymTrajInfo(TrajInfo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Cost = 0
        self.TrajRewards = []
        self.TrajCosts = []
        self.TrajRDists = []
        self.TrajCDists = []
        self.TrajRValues = []
        self.TrajCValues = []
        self.TrajRemainCosts = []
        self.TrajRemainDisCosts = []
        self.consecutive_costs = []
        self.curr_consecutive_cost = 0
        self.prev_cost = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        cost = getattr(env_info, "cost", 0)
        remain_cost = getattr(env_info, "remain_cost", 0)
        remain_discounted_cost = getattr(env_info, "remain_discounted_cost", 0)

        agent_value = getattr(agent_info, "value", 0)
        r_dist = getattr(agent_value, "r_dist", 0)
        c_dist = getattr(agent_value, "c_dist", 0)

        r_dist = r_dist.numpy() if torch.is_tensor(r_dist) else r_dist
        c_dist = c_dist.numpy() if torch.is_tensor(c_dist) else c_dist

        r_value = agent_value["value"] if hasattr(agent_value, "value") else np.mean(r_dist)
        c_value = agent_value["c_value"] if hasattr(agent_value, "c_value") else np.mean(c_dist)

        r_value = r_value.numpy() if torch.is_tensor(r_value) else r_value
        c_value = c_value.numpy() if torch.is_tensor(c_value) else c_value

        self.Cost += cost
        self.TrajRewards.append(reward)
        self.TrajCosts.append(cost)
        self.TrajRDists.append(r_dist)
        self.TrajCDists.append(c_dist)
        self.TrajRValues.append(r_value)
        self.TrajCValues.append(c_value)
        self.TrajRemainCosts.append(remain_cost)
        self.TrajRemainDisCosts.append(remain_discounted_cost)

        self.curr_consecutive_cost += cost
        if self.prev_cost and not cost:
            self.consecutive_costs.append(self.curr_consecutive_cost)
            self.curr_consecutive_cost = 0
        self.prev_cost = cost

    def terminate(self, observation):
        if self.curr_consecutive_cost:
            self.consecutive_costs.append(self.curr_consecutive_cost)
        precision = 3
        m = 10 ** precision
        logger.log(f'R: {round(self.Return * m)/m} \t| C: {round(self.Cost * m)/m} \t| Costs: {self.consecutive_costs}', with_prefix=False, with_timestamp=False)

        del self.NonzeroRewards
        del self.prev_cost
        del self.curr_consecutive_cost
        del self.consecutive_costs

        return super().terminate(observation)


class SafetyGymTrajTracer(object):
    def __init__(self, log_dir, log_interval=5000, n_trace_per_interval=100):
        self._log_dir = log_dir
        self._log_interval = log_interval
        self._n_trace_per_interval = n_trace_per_interval
        self._log_name_base = 'TrajTrace'
        self._task = None

        self._traj_num = 0
        self._traj_trace = dict()

    def set_traj_num(self, i):
        self._traj_num = i

    def get_pos(self, env):
        pos = {
            'goal': env.goal_pos[:2].copy(),
            'robot': env.world.robot_pos()[:2].copy()
        }
        if self._task == 'push':
            pos['box'] = env.box_pos[:2].copy()

        for i, p in enumerate(env.hazards_pos):
            pos[f'hazard{i}'] = p[:2].copy()
        for i, p in enumerate(env.walls_pos):
            pos[f'wall{i}'] = p[:2].copy()
        for i, p in enumerate(env.hazards_pos):
            pos[f'hazard{i}'] = p[:2].copy()
        for i, p in enumerate(env.vases_pos):
            pos[f'vase{i}'] = p[:2].copy()
        for i, p in enumerate(env.gremlins_obj_pos):
            pos[f'gremlin{i}obj'] = p[:2].copy()
        for i, p in enumerate(env.pillars_pos):
            pos[f'pillar{i}'] = p[:2].copy()
        for i, p in enumerate(env.buttons_pos):
            pos[f'button{i}'] = p[:2].copy()
        return pos

    def reset(self, env):
        self._task = env.task
        world_config = env.world_config_dict
        map_size = env.placements_extents


        self._traj_trace = {
            'map': {
                'name': 'map',
                'size': map_size.copy(),
                'group': -1,
            },
            'robot': {
                'name': 'robot',
                'pos': [world_config['robot_xy'].copy()],
                # 'rot': [world_config['robot_rot']],
                'reward': [0.],
                'cost': [0.],
            },
            'geoms': dict(),
            'objects': dict(),
        }
        for obj in ['geoms', 'objects']:
            for k, v in world_config[obj].items():
                if isinstance(v, dict) and bool(v):
                    d = dict()
                    for k1, v1 in v.items():
                        if k1 == 'pos':
                            d[k1] = [v1.copy()[:2]]
                        else:
                            d[k1] = v1
                    self._traj_trace[obj][k] = d

        # print('self._traj_trace: ')
        # print_dict(self._traj_trace)

    def step(self, env, reward, cost, done, **kwargs):
        if self._traj_num > self._log_interval - 1 and self._traj_num % self._log_interval >= 0 and self._traj_num % self._log_interval < self._n_trace_per_interval:
            # For Robot
            pos = self.get_pos(env)
            # print('robot_xy: ', pos['robot'])
            self._traj_trace['robot']['pos'].append(pos['robot'])
            self._traj_trace['robot']['reward'].append(reward)
            self._traj_trace['robot']['cost'].append(cost)

            # For Geoms and Objects
            for obj in ['geoms', 'objects']:
                for k, v in self._traj_trace[obj].items():
                    prev_pos = v['pos'][-1]
                    curr_pos = pos[k]
                    if np.linalg.norm(prev_pos - curr_pos) > 1e-4:
                        self._traj_trace[obj][k]['pos'].append(curr_pos)
            if done:
                self.save(self._log_dir, self._traj_num)

        if done:
            self._traj_num += 1

    def save(self, log_dir, traj_num):
        filename = os.path.join(log_dir, self._log_name_base + f'_{traj_num}.pkl')
        data = {
            'task': self._task,
            'traj_num': traj_num,
            'traj_trace': self._traj_trace,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saving {filename} is successfully done!')

    def load(self, log_dir, traj_num):
        filename = os.path.join(log_dir, self._log_name_base + f'_{traj_num}.pkl')
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self._task = data['task']
        self._traj_num = data['traj_num']
        self._traj_trace = data['traj_trace']
        print(f'Loading {filename} is successfully done!')
        print('self._traj_trace: ')
        print_dict(self._traj_trace)

    def render_trace(self, log_dir, traj_num):
        raise NotImplementedError
