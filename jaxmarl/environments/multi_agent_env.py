"""
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

import jax
from typing import Dict
# Chex is a library of utilities for helping to write reliable JAX code.
import chex
from functools import partial
from flax import struct
from typing import Tuple
from jaxmarl.environments import spaces


# Create a class that can be passed to functional transformations.
@struct.dataclass
class State:
    done: chex.Array
    step: int


class MultiAgentEnv(object):
    """Jittable abstract base class for all jaxmarl Environments."""

    def __init__(
        self,
        num_agents: int,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used
        to set array dimensions
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict()

    # '@partial' is a function in Python's functools module that allows you to
    # partially apply another function with specific arguments.
    # In this case, '@partial(jax.jit, static_argnums=(0, ))' is using @partial
    # to partially apply the jit decorator with the argument
    # static_argnums=(0, ).
    # static_argnums=(0, ) specifies that the first argument of the decorated
    # function (self in this case) is treated as a static argument during JIT
    # compilation.
    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, ))
    def step(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, spaces.Dict]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool],
               Dict]:
        """Performs step transitions in the environment."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(
            key, state, actions)

        obs_re, states_re = self.reset(key_reset)

        # Auto-reset environment based on termination
        # The jax.tree_map() function is a utility for recursively applying a
        # function to each leaf of a nested data structure, such as a nested
        # tuple or dictionary.
        # Depending on whether the current episode is done, return different
        # states and obs
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re,
            states_st)
        obs = jax.tree_map(lambda x, y: jax.lax.select(dones["__all__"], x, y),
                           obs_re, obs_st)
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, spaces.Dict]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool],
               Dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state."""
        raise NotImplementedError

    def observation_space(self, agent: str):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """
        Returns a dictionary with agent classes, used in environments with
        hetrogenous agents.

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError
