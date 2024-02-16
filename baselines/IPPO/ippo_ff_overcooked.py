""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
# from jax_tqdm import scan_tqdm
import time

import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(64,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim,
                              kernel_init=orthogonal(0.01),
                              bias_init=constant(0.0))(actor_mean)
        # print(type(actor_mean))
        pi_1 = distrax.Categorical(logits=actor_mean[..., :6])
        pi_2 = distrax.Categorical(logits=actor_mean[..., 6:9])
        pi_3 = distrax.Categorical(logits=actor_mean[..., 9:])

        critic = nn.Dense(64,
                          kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64,
                          kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1,
                          kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)

        return pi_1, pi_2, pi_3, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def flatten_obs(game_env, obs_dict):
    agents = game_env.agents
    flattened_obs_dict = {}
    for agent in agents:
        obs_arr = obs_dict[agent]['obs'].flatten()
        msg_arr = jnp.concatenate(obs_dict[agent]['msgs'])
        combined_arr = jnp.concatenate([obs_arr, msg_arr])
        flattened_obs_dict[agent] = combined_arr

    return flattened_obs_dict


def get_rollout(train_state, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)

    num_actions = env.action_dim
    network = ActorCritic(num_actions, activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    obs_shape_list = env.observation_space().shape
    init_x = jnp.array([])
    for shape in obs_shape_list:
        temp_init_x = jnp.zeros(shape).flatten()
        init_x = jnp.concatenate([init_x, temp_init_x])

    # Initialize parameters using a PRNGKey and dummy input data
    network.init(key_a, init_x)
    network_params = train_state.params

    done = False
    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_0_1, key_0_2, key_0_3, key_1_1, key_1_2, key_1_3, key_s = \
            jax.random.split(key, 8)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()
        # obs = {k: v.flatten() for k, v in obs.items()}
        obs = flatten_obs(env, obs)

        pi_0_1, pi_0_2, pi_0_3, _ = network.apply(network_params,
                                                  obs["agent_0"])
        pi_1_1, pi_1_2, pi_1_3, _ = network.apply(network_params,
                                                  obs["agent_1"])

        actions = {
            "agent_0":
            jnp.array([
                pi_0_1.sample(seed=key_0_1),
                pi_0_2.sample(seed=key_0_2),
                pi_0_3.sample(seed=key_0_3)
            ],
                      dtype=jnp.uint32),
            "agent_1":
            jnp.array([
                pi_1_1.sample(seed=key_1_1),
                pi_1_2.sample(seed=key_1_2),
                pi_1_3.sample(seed=key_1_3)
            ],
                      dtype=jnp.uint32)
        }
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def non_obs_batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def obs_batchify(x: dict, agent_list, num_actors):
    # alice_obs has shape (num of environments, 4, 5, 26)
    alice_obs = jnp.array(x[agent_list[0]]['obs'])
    # alice_msgs has shape (num of environments, 2, 16)
    alice_msgs = jnp.array(x[agent_list[0]]['msgs'])
    bob_obs = jnp.array(x[agent_list[1]]['obs'])
    bob_msgs = jnp.array(x[agent_list[1]]['msgs'])

    num_rows = int(num_actors / 2)
    alice_array = jnp.concatenate([
        alice_obs.reshape((num_rows, -1)),
        alice_msgs.reshape((num_rows, -1))
    ],
                                  axis=-1)
    bob_array = jnp.concatenate(
        [bob_obs.reshape((num_rows, -1)),
         bob_msgs.reshape((num_rows, -1))],
        axis=-1)

    # print(jnp.concatenate([alice_array, bob_array]).shape)
    return jnp.concatenate([alice_array, bob_array])


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] //
                             config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ACTORS"] * config["NUM_STEPS"] //
                                config["NUM_MINIBATCHES"])

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count //
                      (config["NUM_MINIBATCHES"] *
                       config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        num_actions = env.action_dim
        network = ActorCritic(num_actions, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)

        obs_shape_list = env.observation_space().shape
        init_x = jnp.array([])
        for shape in obs_shape_list:
            temp_init_x = jnp.zeros(shape).flatten()
            init_x = jnp.concatenate([init_x, temp_init_x])

        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5))

        # In Flax, TrainState is a data structure that encapsulates the state
        # of a training process for a neural network model. It contains various
        # pieces of information related to training, such as the model
        # parameters, optimizer state, learning rate schedule, and other
        # metadata.
        # params: the parameters to be updated by tx and used by apply_fn
        # tx: an Optax gradient transformation
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, ))(reset_rng)

        # TRAIN LOOP
        # @scan_tqdm(config["NUM_UPDATES"])
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng_1, _rng_2, _rng_3 = jax.random.split(rng, 4)

                obs_batch = obs_batchify(last_obs, env.agents,
                                         config["NUM_ACTORS"])

                pi_1, pi_2, pi_3, value = network.apply(
                    train_state.params, obs_batch)
                action_1 = pi_1.sample(seed=_rng_1)
                log_prob_1 = pi_1.log_prob(action_1)
                action_2 = pi_2.sample(seed=_rng_2)
                log_prob_2 = pi_2.log_prob(action_2)
                action_3 = pi_3.sample(seed=_rng_3)
                log_prob_3 = pi_3.log_prob(action_3)

                env_act_1 = unbatchify(action_1, env.agents,
                                       config["NUM_ENVS"], env.num_agents)
                env_act_1 = {k: v.flatten() for k, v in env_act_1.items()}
                env_act_2 = unbatchify(action_2, env.agents,
                                       config["NUM_ENVS"], env.num_agents)
                env_act_2 = {k: v.flatten() for k, v in env_act_2.items()}
                env_act_3 = unbatchify(action_3, env.agents,
                                       config["NUM_ENVS"], env.num_agents)
                env_act_3 = {k: v.flatten() for k, v in env_act_3.items()}

                env_act = {
                    env.agents[0]:
                    jnp.array([
                        env_act_1[env.agents[0]], env_act_2[env.agents[0]],
                        env_act_3[env.agents[0]]
                    ],
                              dtype=jnp.uint32).transpose(),
                    env.agents[1]:
                    jnp.array([
                        env_act_1[env.agents[1]], env_act_2[env.agents[1]],
                        env_act_3[env.agents[1]]
                    ],
                              dtype=jnp.uint32).transpose()
                }

                action = jnp.array([action_1, action_2, action_3],
                                   dtype=jnp.uint32).transpose()
                log_prob = log_prob_1 + log_prob_2 + log_prob_3

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                # 'jax.tree_map()' is a JAX utility function used for
                # recursively mapping a function over a tree-like data
                # structure. It applies the provided function to each leaf
                # element of the input tree while preserving the overall
                # structure of the tree.
                info = jax.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    non_obs_batchify(done, env.agents,
                                     config["NUM_ACTORS"]).squeeze(), action,
                    value,
                    non_obs_batchify(reward, env.agents,
                                     config["NUM_ACTORS"]).squeeze(), log_prob,
                    obs_batch, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            # 'scan' is a JAX function that enables efficient looping over
            # array elements with carryover values.
            # f: a Python function to be scanned
            # init: an initial loop carry value
            # xs: the value of type [a] over which to scan along the leading
            #     axis
            # length: optional integer specifying the number of loop iterations
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state,
                                                    None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = obs_batchify(last_obs, env.agents,
                                          config["NUM_ACTORS"])
            _, _, _, last_val = network.apply(train_state.params,
                                              last_obs_batch)

            def _calculate_gae(traj_batch, last_val):

                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (
                        1 - done) - value
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"] *
                           (1 - done) * gae)
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):

                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi_1, pi_2, pi_3, value = network.apply(
                            params, traj_batch.obs)
                        log_prob_1 = pi_1.log_prob(traj_batch.action[:, 0])
                        log_prob_2 = pi_2.log_prob(traj_batch.action[:, 1])
                        log_prob_3 = pi_3.log_prob(traj_batch.action[:, 2])
                        log_prob = log_prob_1 + log_prob_2 + log_prob_3

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped -
                                                          targets)
                        value_loss = (0.5 * jnp.maximum(
                            value_losses, value_losses_clipped).mean())

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * gae)
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi_1.entropy().mean() + pi_2.entropy().mean(
                        ) + pi_3.entropy().mean()

                        total_loss = (loss_actor +
                                      config["VF_COEF"] * value_loss -
                                      config["ENT_COEF"] * entropy)
                        return total_loss, (value_loss, loss_actor, entropy)

                    # We need to provide jax.value_and_grad() with a function
                    # that you want to differentiate. This function should take
                    # one or more input arguments and return a scalar value.
                    # jax.value_and_grad() returns a new function that, when
                    # called with the same input arguments as the original
                    # function, computes both the value of the original
                    # function and its gradient with respect to the input
                    # arguments.
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch,
                                                advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config[
                    "NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size, ) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] +
                                          list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch,
                                                       train_state,
                                                       minibatches)
                update_state = (train_state, traj_batch, advantages, targets,
                                rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state,
                                                   None,
                                                   config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None,
                                            config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None,
            config_path="config",
            config_name="ippo_ff_overcooked")
def main(config):
    # Convert an OmegaConf object (typically a Config object) into a native
    # Python container.
    start_time = time.time()
    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]
                                                        ["layout"]]
    rng = jax.random.PRNGKey(30)
    num_seeds = 20

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(config)))
        rngs = jax.random.split(rng, num_seeds)
        out = train_jit(rngs)

    end_time = time.time()
    print('Time: ', end_time - start_time)

    print('** Saving Results **')
    filename = f'{config["ENV_NAME"]}_cramped_room_new_new'
    rewards = out["metrics"]["returned_episode_returns"].mean(-1).reshape(
        (num_seeds, -1))
    reward_mean = rewards.mean(0)  # mean
    reward_std = rewards.std(0) / np.sqrt(num_seeds)  # standard error

    plt.plot(reward_mean)
    plt.fill_between(range(len(reward_mean)),
                     reward_mean - reward_std,
                     reward_mean + reward_std,
                     alpha=0.2)
    # compute standard error
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'{filename}.png')

    # animate first seed
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")


if __name__ == "__main__":
    main()
