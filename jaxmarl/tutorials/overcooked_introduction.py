"""
Short introduction to running the Overcooked environment and visualising it using random actions.
"""

import jax
from jaxmarl import make
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts, layout_grid_to_dict
from jaxmarl.environments.overcooked.overcooked import Actions, Msg_Template
import time
import random
import jax.numpy as jnp
import numpy as np


def gen_random_actions(env, key):
    # Sample random actions
    key_a = jax.random.split(key, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }

    return actions


# Parameters + random keys
max_steps = 100
# First generate a random key
key = jax.random.PRNGKey(0)
# Split this key into 3 keys. Each key generated by jax.random.split() is
# independent and can be used to generate random numbers.
key, key_r, key_a = jax.random.split(key, 3)

# Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
layout = overcooked_layouts["cramped_room"]

# Or make your own!
# custom_layout_grid = """
# WWOWW
# WA  W
# B P X
# W  AW
# WWOWW
# """
# layout = layout_grid_to_dict(custom_layout_grid)

# Instantiate environment
env = make('overcooked', layout=layout, max_steps=max_steps)

obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

state_seq = []
for _ in range(max_steps):
    state_seq.append(state)
    # Iterate random keys and sample actions
    key, key_s, key_a = jax.random.split(key, 3)

    actions = gen_random_actions(env, key_a)

    # Step environment
    obs, state, rewards, dones, infos = env.step(key_s, state, actions)

# viz = OvercookedVisualizer()

# # Render to screen
# for s in state_seq:
#     viz.render(env.agent_view_size, s, highlight=False)
#     time.sleep(0.25)

# # # Or save an animation
# # viz.animate(state_seq, agent_view_size=5, filename='animation.gif')
