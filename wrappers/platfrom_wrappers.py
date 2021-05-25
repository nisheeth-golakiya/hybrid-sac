import copy
import gym
import numpy as np
from gym.spaces import Tuple, Box


class ScaledStateWrapper(gym.ObservationWrapper):
    """
    Scales the observation space to [-1,1]
    """
    def __init__(self, env):
        super(ScaledStateWrapper, self).__init__(env)
        obs = env.observation_space
        self.compound = False
        self.low = None
        self.high = None
        print(type(obs))
        print(obs)
        if isinstance(obs, gym.spaces.Box):
            self.low = env.observation_space.low
            self.high = env.observation_space.high
            self.observation_space = gym.spaces.Box(low=-np.ones(self.low.shape),
                                                    high=np.ones(self.high.shape),
                                                    dtype=np.float32)
        elif isinstance(obs, Tuple):
            self.low = obs.spaces[0].low
            self.high = obs.spaces[0].high
            assert len(obs.spaces) == 2 and isinstance(obs.spaces[1], gym.spaces.Discrete)
            self.observation_space = Tuple((gym.spaces.Box(low=-np.ones(self.low.shape),
                                                           high=np.ones(self.high.shape),
                                                           dtype=np.float32), obs.spaces[1]))
            self.compound = True
        else:
            raise Exception("Unsupported observation space type: %s" % self.observation_space)

    def scale_state(self, state):
        state = 2. * (state - self.low) / (self.high - self.low) - 1.
        return state

    def _unscale_state(self, scaled_state):
        state = (self.high - self.low) * (scaled_state + 1.) / 2. + self.low
        return state

    def observation(self, obs):
        if self.compound:
            state, steps = obs
            ret = (self.scale_state(state), steps)
        else:
            ret = self.scale_state(obs)
        return ret


class ScaledParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space must be flattened!
    Tuple((
        Discrete(n),
        Box(c_1),
        Box(c_2),
        ...
        Box(c_n)
        )
    """
    def __init__(self, env):
        super(ScaledParameterisedActionWrapper, self).__init__(env)
        self.old_as = env.action_space
        self.num_actions = self.old_as.spaces[0].n
        self.high = [self.old_as.spaces[i].high for i in range(1, self.num_actions + 1)]
        self.low = [self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)]
        self.range = [self.old_as.spaces[i].high - self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)]
        new_params = [  # parameters
            Box(-np.ones(self.old_as.spaces[i].low.shape), np.ones(self.old_as.spaces[i].high.shape), dtype=np.float32)
            for i in range(1, self.num_actions + 1)
        ]
        self.action_space = Tuple((
            self.old_as.spaces[0],  # actions
            *new_params,
        ))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.
        :param action:
        :return:
        """
        action = copy.deepcopy(action)
        p = action[0]
        action[1][p] = self.range[p] * (action[1][p] + 1) / 2. + self.low[p]
        return action


class PlatformFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(PlatformFlattenedActionWrapper, self).__init__(env)
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))))

    def action(self, action):
        return action
