from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
# 'FrozenDict' is a class used to represent immutable dictionaries, which can
# be particularly useful for holding model parameters and configuration
# settings
from flax.core.frozen_dict import FrozenDict

from jaxmarl.environments.overcooked.common import (OBJECT_TO_INDEX,
                                                    COLOR_TO_INDEX,
                                                    OBJECT_INDEX_TO_VEC,
                                                    DIR_TO_VEC,
                                                    make_overcooked_map)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as \
    layouts
from transformers import RobertaTokenizer


class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5
    done = 6


class Msg_Template():
    templates = ['', 'I am going to ', 'I need ']
    options = [
        '', 'onion', 'onion pile', 'plate', 'plate pile', 'goal', 'pot', 'dish'
    ]


@struct.dataclass
class State:
    agent_pos: chex.Array
    agent_dir: chex.Array
    agent_dir_idx: chex.Array
    agent_inv: chex.Array
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    agent_comm: chex.Array
    time: int
    terminal: bool


# Pot status indicated by an integer, which ranges from 23 to 0
# 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_EMPTY_STATUS = 23
# 3 onions. Below this status, pot is cooking, and status acts like a countdown
# timer.
POT_FULL_STATUS = 20
POT_READY_STATUS = 0
# A pot has at most 3 onions. A soup contains exactly 3 onions.
MAX_ONIONS_IN_POT = 3

# When this many time steps remain, the urgency layer is flipped on
URGENCY_CUTOFF = 40
DELIVERY_REWARD = 20


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""

    def __init__(self,
                 layout=FrozenDict(layouts["cramped_room"]),
                 random_reset: bool = False,
                 max_steps: int = 400,
                 tokenizer=RobertaTokenizer.from_pretrained('roberta-base')):
        # Sets self.num_agents to 2
        super().__init__(num_agents=2)

        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        self.height = layout["height"]
        self.width = layout["width"]
        # TODO: Need to change
        self.obs_shape = (self.width, self.height, 26)

        # Hard coded. Only affects map padding -- not observations.
        self.agent_view_size = 5
        self.layout = layout
        self.agents = ["agent_0", "agent_1"]

        # TODO: Need to change
        self.action_set = jnp.array([
            Actions.right,
            Actions.down,
            Actions.left,
            Actions.up,
            Actions.stay,
            Actions.interact,
        ])
        self.tokenizer = tokenizer
        self.max_length = 16
        self.msg_matrix = self._tokenize_msgs()
        self.empty_msg = self.msg_matrix[0, 0, :]

        self.random_reset = random_reset
        self.max_steps = max_steps

    def _tokenize_msgs(self) -> chex.Array:
        """
        Produce a matrix of size:
        (number of templates, number of options, length of text embedding)

        The inner most array represents the text embeddings of a certain
        message composed of a template + an option.
        """
        msg_matrix = []
        for i in range(len(Msg_Template.templates)):
            row = []
            for j in range(len(Msg_Template.options)):
                text = Msg_Template.templates[i] + Msg_Template.options[j]
                encoded_text = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='np')
                row.append(encoded_text['input_ids'][0])
            row = jnp.array(np.array(row))
            msg_matrix.append(row)
        msg_matrix = jnp.array(np.array(msg_matrix))

        return msg_matrix

    def step_env(
        self, key: chex.PRNGKey, state: State,
        actions: Dict[str, spaces.MultiDiscrete]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool],
               Dict]:
        """
        Perform single timestep state transition.

        Args:
        - key: a list of two keys
        - state: a State object (defined in this module)
        - actions: a mapping from agent name to action

        Returns:
        - observations
        - states
        - rewards
        - whether done
        """
        # acts is an array containing the action of each agent
        # '.take()' takes elements from an array along an axis.
        acts = self.action_set.take(
            indices=jnp.array([actions["agent_0"][0], actions["agent_1"][0]]))

        # msgs_idx_transpose = jnp.transpose(
        #     jnp.vstack([actions["agent_0"][1:], actions["agent_1"][1:]]))
        # msgs = self.msg_matrix[msgs_idx_transpose[0], msgs_idx_transpose[1], :]
        msgs = jnp.vstack([actions["agent_0"][1:], actions["agent_1"][1:]])

        # Execute the agents' actions in the environment
        state, reward = self.step_agents(key, state, acts, msgs)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        # 'lax.stop_gradient()' excludes a variable from participating in the
        # gradient computation during automatic differentiation
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {},
        )

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """
        Reset environment state based on 'self.random_reset'

        If True, everything is randomized, including agent inventories and
        positions, pot states and items on counters
        If False, only resample agent orientations

        In both cases, the environment layout is determined by 'self.layout'

        Args:
        - key: a list of two keys

        Returns:
        - initial observation (no grad)
        - initial state (no grad)
        """

        # Whether to fully randomize the start state
        random_reset = self.random_reset
        layout = self.layout

        h = self.height
        w = self.width
        num_agents = self.num_agents
        # np.prod() returns the product of array elements over a given axis
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")

        # 'jnp.zeros_like()' returns an array of zeros with the same shape and
        # type as a given array
        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        # 'wall_map' is a 2D array of size h by w. An entry is True if there's
        # a wall and false otherwise
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent position + direction
        key, subkey = jax.random.split(key)
        # the 'p' argument below takes an array, providing the probabilities
        # associated with each entry in a
        agent_idx = jax.random.choice(
            subkey,
            all_pos,
            shape=(num_agents, ),
            p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32),
            replace=False)

        # Replace with fixed layout if applicable. Also randomize if agent
        # position not provided
        agent_idx = random_reset * agent_idx + (1 - random_reset) * layout.get(
            "agent_idx", agent_idx)
        # The position of each agent is as follows: the grid number along the
        # positive x axis, and the grid number along the negative y axis,
        # with the top left grid having coordinate (0, 0). All valid
        # coordinates are positive
        agent_pos = jnp.array(
            [agent_idx % w, agent_idx // w],
            dtype=jnp.uint32).transpose()  # dim = n_agents x 2
        occupied_mask = occupied_mask.at[agent_idx].set(1)

        # Randomize agent direction
        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey,
                                          jnp.arange(len(DIR_TO_VEC),
                                                     dtype=jnp.int32),
                                          shape=(num_agents, ))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get()  # dim = n_agents x 2

        # Keep track of empty counter space (i.e. onion pile and plate pile)
        empty_table_mask = jnp.zeros_like(all_pos)
        # The wall is empty counter space
        empty_table_mask = empty_table_mask.at[wall_idx].set(1)

        # Delivery locations are not empty counter space
        goal_idx = layout.get("goal_idx")
        goal_pos = jnp.array([goal_idx % w, goal_idx // w],
                             dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        # The locations of onion piles are not empty counter space
        onion_pile_idx = layout.get("onion_pile_idx")
        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w],
                                   dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        # The locations of plate piles are not empty counter space
        plate_pile_idx = layout.get("plate_pile_idx")
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w],
                                   dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        # Pot locations are not empty counter space
        pot_idx = layout.get("pot_idx")
        pot_pos = jnp.array([pot_idx % w, pot_idx // w],
                            dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        # Randomize pot(s) status
        key, subkey = jax.random.split(key)
        # Pot status is determined by a number between 0 (inclusive) and 24
        # (exclusive)
        # 23 corresponds to an empty pot (default)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0], ), 0, 24)
        pot_status = pot_status * random_reset + (1 - random_reset) * jnp.ones(
            (pot_idx.shape[0])) * 23

        onion_pos = jnp.array([])
        plate_pos = jnp.array([])
        dish_pos = jnp.array([])

        maze_map = make_overcooked_map(wall_map,
                                       goal_pos,
                                       agent_pos,
                                       agent_dir_idx,
                                       plate_pile_pos,
                                       onion_pile_pos,
                                       pot_pos,
                                       pot_status,
                                       onion_pos,
                                       plate_pos,
                                       dish_pos,
                                       pad_obs=True,
                                       num_agents=self.num_agents,
                                       agent_view_size=self.agent_view_size)

        # Reset agents' inventory (empty by default, can be randomized)
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([
            OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
            OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']
        ])
        random_agent_inv = jax.random.choice(subkey,
                                             possible_items,
                                             shape=(num_agents, ),
                                             replace=True)
        agent_inv = random_reset * random_agent_inv + \
            (1-random_reset) * jnp.array(
                [OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['empty']])

        # agent_comm = jnp.vstack([self.empty_msg, self.empty_msg])
        empty_msg = jnp.array([0, 0], dtype=jnp.uint32)
        agent_comm = jnp.array([empty_msg, empty_msg])

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            agent_comm=agent_comm,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs(self, state: State) -> Dict[str, spaces.Dict]:
        """
        Return a full observation, of size (height x width x n_layers), where
        n_layers = 26, together with the communication msgs.
        Layers are of shape (height x width) and  are binary (0/1) except where
        indicated otherwise.
        The obs is very sparse (most elements are 0), which prob. contributes
        to generalization problems in Overcooked.
        A v2 of this environment should have much more efficient observations,
        e.g. using item embeddings.

        The list of channels is below. Agent-specific layers are ordered so
        that an agent perceives its layers first.
        Env layers are the same (and in same order) for both agents.

        Agent positions :
        0. position of agent i (1 at agent loc, 0 otherwise)
        1. position of agent (1-i)

        Agent orientations :
        2-5. agent_{i}_orientation_0 to agent_{i}_orientation_3 (layers are
        entirely zero except for the one orientation layer that matches the
        agent orientation. That orientation has a single 1 at the agent
        coordinates.)
        6-9. agent_{i-1}_orientation_{dir}

        Static env positions(1 where object of type X is located, 0 otherwise):
        10. pot locations
        11. counter locations (table)
        12. onion pile locations
        13. tomato pile locations (tomato layers are included for consistency,
        but this env does not support tomatoes)
        14. plate pile locations
        15. delivery locations (goal)

        Pot and soup specific layers. These are non-binary layers:
        16. number of onions in pot (0,1,2,3) for elements corresponding to pot
        locations. Nonzero only for pots that have NOT started cooking yet.
        When a pot starts cooking (or is ready), the corresponding element is
        set to 0
        17. number of tomatoes in pot.
        18. number of onions in soup (0,3) for elements corresponding to either
        a cooking/done pot or to a soup (dish) ready to be served. This is a
        useless feature since all soups have exactly 3 onions, but it made
        sense in the full Overcooked where recipes can be a mix of tomatoes and
        onions
        19. number of tomatoes in soup
        20. pot cooking time remaining. [19 -> 1] for pots that are cooking. 0
        for pots that are not cooking or done
        21. soup done. (Binary) 1 for pots done cooking and for locations
        containing a soup (dish). O otherwise.

        Variable env layers (binary):
        22. plate locations
        23. onion locations
        24. tomato locations

        Urgency:
        25. Urgency. The entire layer is 1 there are 40 or fewer remaining time
        steps. 0 otherwise
        """

        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = self.obs_shape[2]
        padding = (state.maze_map.shape[0] - height) // 2

        # Get the maze map without paddings and only showing the item id
        # maze_map is a 2D array whose number of rows equals the height of the
        # layout and number columns equals to width of the layout
        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]

        # A 2D array of 0s and 1s indicating if a location has soup
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"],
                             dtype=jnp.uint8)

        # A 2D array of 0s and 1s indicating if a location has a pot
        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"],
                                  dtype=jnp.uint8)
        # A 2D array indicating the status of the pot at a location
        pot_status = state.maze_map[padding:-padding, padding:-padding,
                                    2] * pot_loc_layer
        # A 2D array indicating the number of onions in a non-cooking or not
        # done pot at a location
        onions_in_pot_layer = jnp.minimum(
            POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (
                pot_status >= POT_FULL_STATUS
            )  # 0/1/2/3, as long as not cooking or not done
        # A 2D array indicating the number of onions in a cooking/done pot or
        # soup. (3 for locations with a cooking/done pot or soup and 0 o.w.)
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc
        # A 2D array indicating the amount of cooking time left in the pot
        # (if location does not have a pot, the value is 0)
        pot_cooking_time_layer = pot_status * (pot_status < POT_FULL_STATUS
                                               )  # Timer: 19 to 0
        # A 2D array indicating the locations of soups that are ready, either
        # plated or not
        soup_ready_layer = pot_loc_layer * (pot_status
                                            == POT_READY_STATUS) + soup_loc
        # The entire layer is 1 there are {40} or fewer remaining time steps.
        # 0 otherwise
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * (
            (self.max_steps - state.time) < URGENCY_CUTOFF)

        # 2 2D arrays indicating the locations of the locations of the two
        # agents
        agent_pos_layers = jnp.zeros((2, height, width), dtype=jnp.uint8)
        agent_pos_layers = agent_pos_layers.at[0, state.agent_pos[0, 1],
                                               state.agent_pos[0, 0]].set(1)
        agent_pos_layers = agent_pos_layers.at[1, state.agent_pos[1, 1],
                                               state.agent_pos[1, 0]].set(1)

        # Add agent inventory. This works because loose items and agent cannot
        # overlap.
        # 'state.agent_inv' is an array of length number of agents.
        # Expanding it along axis=(1, 2) results in a 3D array of dimension:
        # number of agents x 1 x 1.
        # Then, we broadcast the expanded array to 'agent_pos_layers'.
        agent_inv_items = jnp.expand_dims(state.agent_inv,
                                          (1, 2)) * agent_pos_layers
        # In the environment map, set the entries of locations where there is
        # an agent to the agent's inventory.
        maze_map = jnp.where(jnp.sum(agent_pos_layers, 0),
                             agent_inv_items.sum(0), maze_map)
        # If an agent is carrying soup, add that to the soup ready layer.
        soup_ready_layer = soup_ready_layer \
                           + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * jnp.sum(agent_pos_layers,0)
        # Similarly, if an agent is carrying soup, add the number of onions to
        # its corresponding layer.
        onions_in_soup_layer = onions_in_soup_layer \
                               + (jnp.sum(agent_inv_items,0) == OBJECT_TO_INDEX["dish"]) * 3 * jnp.sum(agent_pos_layers,0)

        env_layers = [
            jnp.array(maze_map == OBJECT_TO_INDEX["pot"],
                      dtype=jnp.uint8),  # Channel 10
            jnp.array(maze_map == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion_pile"],
                      dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomato pile
            jnp.array(maze_map == OBJECT_TO_INDEX["plate_pile"],
                      dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["goal"],
                      dtype=jnp.uint8),  # 15
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),  # 20
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes
            urgency_layer,  # 25
        ]

        # Agent related layers
        # There are 8 'agent_direction_layers' because we need 4 direction
        # layers per agent.
        agent_direction_layers = jnp.zeros((8, height, width), dtype=jnp.uint8)
        # Element wise addition to index the relevant direction layers for the
        # agents.
        dir_layer_idx = state.agent_dir_idx + jnp.array([0, 4])
        agent_direction_layers = agent_direction_layers.at[
            dir_layer_idx, :, :].set(agent_pos_layers)

        # Both agent see their layers first, then the other layer
        alice_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        alice_obs = alice_obs.at[0:2].set(agent_pos_layers)
        alice_obs = alice_obs.at[2:10].set(agent_direction_layers)
        alice_obs = alice_obs.at[10:].set(jnp.stack(env_layers))

        bob_obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)
        bob_obs = bob_obs.at[0].set(agent_pos_layers[1]).at[1].set(
            agent_pos_layers[0])
        bob_obs = bob_obs.at[2:6].set(agent_direction_layers[4:]).at[6:10].set(
            agent_direction_layers[0:4])
        bob_obs = bob_obs.at[10:].set(jnp.stack(env_layers))

        # In 'jnp.transpose()', the axis argument is a tuple or list which
        # contains a permutation of [0,1,…,N-1] where N is the number of axes
        # of a. The i'th axis of the returned array will correspond to the axis
        # numbered axes[i] of the input.
        # Using axis=(1, 2, 0), the shape becomes: number of rows in the env x
        # number of columns in the env x number of channels.
        alice_obs = jnp.transpose(alice_obs, (1, 2, 0))
        bob_obs = jnp.transpose(bob_obs, (1, 2, 0))

        # Each agent sees their msg first
        alice_msgs = jnp.array([state.agent_comm[0], state.agent_comm[1]])
        bob_msgs = jnp.array([state.agent_comm[1], state.agent_comm[0]])

        alice_obs_msgs = {'obs': alice_obs, 'msgs': alice_msgs}
        bob_obs_msgs = {'obs': bob_obs, 'msgs': bob_msgs}

        return {"agent_0": alice_obs_msgs, "agent_1": bob_obs_msgs}

    def step_agents(self, key: chex.PRNGKey, state: State, action: chex.Array,
                    msgs: chex.Array) -> Tuple[State, float]:

        # Update agent position (forward action)
        # 'is_move_action' is an array of size number of agents, and each of
        # its entry is whether the action taken by that agent is a move action
        is_move_action = jnp.logical_and(action != Actions.stay, action
                                         != Actions.interact)
        # First, add a dimension to 'is_move_action' (i.e. if it's [True, True]
        # before, now it's [[True, True]]).
        # Then, transpose the array to make it become [[True], [True]].
        # Has to add a dimension first because transposing a 1D array doesn't
        # change the array.
        is_move_action_transposed = jnp.expand_dims(
            is_move_action, 0).transpose()  # Necessary to broadcast correctly

        # Calculate the forward position of the agent.
        # If the input is a move action, calculate the position the agent would
        # travel to assuming there is no obstacle or another agent in front.
        # The output of 'DIR_TO_VEC[jnp.minimum(action, 3)]' is a 2D array,
        # in which the first dimension is the number of agents and the second
        # dimension contains the vectors representing the direction of the
        # agent's actions.
        # print(state.agent_pos)
        # print(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
        #                 + ~is_move_action_transposed * state.agent_dir)
        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)] \
                        + ~is_move_action_transposed * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            """
            Args:
            - fwd_position: the position of an agent
            - wall_map: a 2D array indicating whether there is a wall at that
                        location
            - goal_pos: a list of goal positions

            Returns:
            - whether the agent is at a wall
            - whether the agent is at a goal
            """
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal: jnp.logical_and(
                pos[0] == goal[0], pos[1] == goal[1])
            fwd_goal = jax.vmap(goal_collision,
                                in_axes=(None, 0))(fwd_position, goal_pos)
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        # 'jax.vmap()' (vectorized map) allows one to automatically vectorize
        # functions, making it efficient to apply them element-wise to arrays
        # or along specified axees. (Vectorizing a function means modifying the
        # function so that it can operate efficiently on arrays or sequences of
        # data, rather than processing individual elements one at a time.)
        # The 'in_axes' parameter is used to specify which axes of the input
        # arguments should be mapped over. In this case, we will apply the
        # vectorized function over the 0th axis for the first parameter and
        # will not vectorize over the second and third parameters.
        # 'fwd_pos_has_wall' is a 1D array, whose ith element indicates whether
        #  there is a wall for the ith agent's position.
        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(
            _wall_or_goal, in_axes=(0, None, None))(fwd_pos, state.wall_map,
                                                    state.goal_pos)

        # 'fwd_pos_blocked' is a 2d array of shape number of agents x 1.
        # An entry in 'fwd_pos_blocked' is true if there is a wall or goal
        # and false otherwise.
        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall,
                                         fwd_pos_has_goal).reshape(
                                             (self.num_agents, 1))

        # 'bounced' is a 2d array of shape number of agents x 1.
        # An entry in 'bounced' is true if the agent needs to be bounced back,
        # which is when the forward position is blocked or the agent doesn't
        # perform a move action.
        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Hardcoded for 2 agents (call them Alice and Bob)!!!
        # Set 'fwd_pos' to the old position if the agent's corresponding bounce
        # entry is true and new position if otherwise.
        # 'agent_pos_prev' has shape number of agents x 2
        agent_pos_prev = jnp.array(state.agent_pos)
        # 'fwd_pos' has shape number of agents x 2
        # If the agent needs to be bounced back, set the forward position to
        # its old position.
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(
            jnp.uint32)

        # Agents can't overlap.
        # 'collision' is a boolean value
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        # 'jnp.where()' is a function that performs element-wise conditional
        # selection on arrays. If 'collision' is true, then we will select the
        # element from state.agent_pos[0]; if not, we will select from
        # fwd_pos[0].
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],  # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            state.agent_pos[1],  # collision and Alice bounced
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        # 'swap_places' is a boolean value
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )

        # If the agents have swapped places, set their positions to their
        # original positions
        alice_pos = jnp.where(~collision * swap_places, state.agent_pos[0],
                              alice_pos)
        bob_pos = jnp.where(~collision * swap_places, state.agent_pos[1],
                            bob_pos)

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        # If the agent didn't move, use the current agent direction index.
        # Otherwise, set the direction to be the same as its action.
        # 'agent_dir_idx' is an array of length number of agents. Each entry of
        # the array is the index of the direction for that agent.
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + \
            is_move_action * action
        # 'agent_dir' is a 2D array of size number of agents x 2. Each 'row' in
        # the 2D array is a direction vector.
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision
        # handling.
        # This matches the original Overcooked
        # Calculate the position in front of the agent so that the agent can
        # potentially interact with that position.
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        # print(maze_map)
        # 'is_interact_action' is an array of size number of agents. Each entry
        # in the array is a boolean indicating whether that agent has performed
        # an interact action.
        is_interact_action = (action == Actions.interact)

        # Compute the effect of interact first, then apply it if needed
        # print(fwd_pos[0])
        candidate_maze_map, alice_inv, alice_reward = self.process_interact(
            maze_map, state.wall_map, fwd_pos[0], state.agent_inv[0])
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        # 'jax.lax.select' functions the same as jnp.where() if the first
        # argument is a boolean. If the first argument is an array, then we
        # will select elements based on each entry in the predicate array.
        maze_map = jax.lax.select(alice_interact, candidate_maze_map, maze_map)
        # 'alice_inv' is an integer that denotes alice's inventory
        alice_inv = jax.lax.select(alice_interact, alice_inv,
                                   state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.)

        candidate_maze_map, bob_inv, bob_reward = self.process_interact(
            maze_map, state.wall_map, fwd_pos[1], state.agent_inv[1])
        maze_map = jax.lax.select(bob_interact, candidate_maze_map, maze_map)
        bob_inv = jax.lax.select(bob_interact, bob_inv, state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.)

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev,
                               agent_idx):
            agent = jnp.array([
                OBJECT_TO_INDEX['agent'],
                COLOR_TO_INDEX['red'] + agent_idx * 2, agent_dir_idx
            ],
                              dtype=jnp.uint8)
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(
            agent_dir_idx, agent_pos, agent_pos_prev,
            jnp.arange(self.num_agents))
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev,
                               padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y,
                               padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= POT_FULL_STATUS)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = is_cooking * not_done * (pot_status - 1) + (
                ~is_cooking) * pot_status  # defaults to zero if done
            return pot.at[-1].set(pot_status)

        # 'state.pot_pos' is a 2D array of pot positions. (I think they are
        # assuming there can only be one pot.)
        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        reward = alice_reward + bob_reward

        return (state.replace(agent_pos=agent_pos,
                              agent_dir_idx=agent_dir_idx,
                              agent_dir=agent_dir,
                              agent_inv=agent_inv,
                              maze_map=maze_map,
                              agent_comm=msgs,
                              terminal=False), reward)

    def process_interact(self, maze_map: chex.Array, wall_map: chex.Array,
                         fwd_pos: chex.Array, inventory: chex.Array):
        """
        Assume agent took interact actions. Result depends on what agent is
        facing and what it is holding.
        """

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        # Get object in front of agent (on the "table")
        maze_object_on_table = maze_map.at[padding + fwd_pos[1],
                                           padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]  # Simple index

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(
            object_on_table == OBJECT_TO_INDEX["plate_pile"],
            object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(
            object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"],
                           object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"])

        # Whether the object in front is counter space that the agent can drop
        # on.
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(),
                                   ~object_is_pot)

        table_is_empty = jnp.logical_or(
            object_on_table == OBJECT_TO_INDEX["wall"],
            object_on_table == OBJECT_TO_INDEX["empty"])

        # Pot status (used if the object is a pot)
        pot_status = maze_object_on_table[-1]

        # Get inventory object, and related booleans
        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        # Interactions with pot. 3 cases: add onion if missing, collect soup if
        # ready, do nothing otherwise
        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status
                  == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (
            pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        # Update pot status and object in inventory
        new_pot_status = \
            case_1 * (pot_status - 1) \
            + case_2 * POT_EMPTY_STATUS \
            + case_3 * pot_status \
            + else_case * pot_status
        new_object_in_inv = \
            case_1 * OBJECT_TO_INDEX["empty"] \
            + case_2 * OBJECT_TO_INDEX["dish"] \
            + case_3 * object_in_inv \
            + else_case * object_in_inv

        # Interactions with onion/plate piles and objects on counter
        # Pickup if: table, not empty, room in inv & object is not something
        # unpickable (e.g. pot or goal)
        successful_pickup = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(
            object_is_pile, object_is_pickable)
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(
            jnp.logical_and(~successful_pickup, ~successful_drop),
            ~successful_delivery)

        # Update object on table
        new_object_on_table = \
            no_effect * object_on_table \
            + successful_delivery * object_on_table \
            + successful_pickup * object_is_pile * object_on_table \
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"] \
            + successful_drop * object_in_inv

        # Update object in inventory
        new_object_in_inv = \
            no_effect * new_object_in_inv \
            + successful_delivery * OBJECT_TO_INDEX["empty"] \
            + successful_pickup * object_is_pickable * object_on_table \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"] \
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"] \
            + successful_drop * OBJECT_TO_INDEX["empty"]

        # Apply inventory update
        inventory = new_object_in_inv

        # Apply changes to maze
        new_maze_object_on_table = \
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status) \
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table] \
            + object_is_agent * maze_object_on_table

        maze_map = maze_map.at[padding + fwd_pos[1], padding +
                               fwd_pos[0], :].set(new_maze_object_on_table)

        # Reward of 20 for a soup delivery
        reward = jnp.array(successful_delivery, dtype=float) * DELIVERY_REWARD
        return maze_map, inventory, reward

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_eval_solved_rate_fn(self):

        def _fn(ep_stats):
            return ep_stats['return'] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set) * len(Msg_Template.templates) * len(
            Msg_Template.options)

    @property
    def action_dim(self) -> int:
        """The number of output nodes we need to output actions."""
        return len(self.action_set) + len(Msg_Template.templates) + len(
            Msg_Template.options)

    def action_space(self, agent_id="") -> spaces.Dict:
        """
        Action space of the environment. Agent_id not used since action_space
        is uniform for all agents
        """
        return spaces.MultiDiscrete([
            len(self.action_set),
            len(Msg_Template.templates),
            len(Msg_Template.options)
        ])

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        obs = spaces.Box(0, 255, self.obs_shape)
        # msgs = spaces.Box(0,
        #                   jax.dtypes.iinfo(jnp.uint32).max,
        #                   (2, self.max_length),
        #                   dtype=jnp.uint32)
        msgs = spaces.Box(0, 7, (2, 2), dtype=jnp.uint32)
        # msgs = spaces.MultiDiscrete(
        #     [len(Msg_Template.templates),
        #      len(Msg_Template.options)])
        return spaces.Dict({'obs': obs, 'msgs': msgs})

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict({
            "agent_pos":
            spaces.Box(0, max(w, h), (2, ), dtype=jnp.uint32),
            "agent_dir":
            spaces.Discrete(4),
            "goal_pos":
            spaces.Box(0, max(w, h), (2, ), dtype=jnp.uint32),
            "maze_map":
            spaces.Box(0,
                       255, (w + agent_view_size, h + agent_view_size, 3),
                       dtype=jnp.uint32),
            # "agent_comm":
            # spaces.Box(0,
            #            jax.dtypes.iinfo(jnp.uint32).max, (2, self.max_length),
            #            dtype=jnp.uint32),
            "agent_comm":
            spaces.Box(0, 7, (2, 2), dtype=jnp.uint32),
            # "agent_comm":
            # spaces.MultiDiscrete(
            #     [len(Msg_Template.templates),
            #      len(Msg_Template.options)]),
            "time":
            spaces.Discrete(self.max_steps),
            "terminal":
            spaces.Discrete(2),
        })

    def max_steps(self) -> int:
        return self.max_steps
