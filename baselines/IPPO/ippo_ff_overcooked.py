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
import time

import matplotlib.pyplot as plt

import os
import shutil
import orbax.checkpoint
from flax.training import orbax_utils

from jax_tqdm import scan_tqdm
# from tqdm import tqdm

# import time
# import datetime
# from jax.experimental import host_callback

# def progress_bar_scan(num_samples, message=None):
#     "Progress bar for a JAX scan"
#     if message is None:
#         message = f"Running for {num_samples} iterations"
#     tqdm_bars = {}

#     if num_samples > 20:
#         print_rate = int(num_samples / 20)
#     else:
#         print_rate = 1  # if you run the sampler for less than 20 iterations
#     remainder = num_samples % print_rate

#     def _define_tqdm(arg, transform):
#         tqdm_bars[0] = tqdm(range(num_samples))
#         tqdm_bars[0].set_description(message, refresh=False)

#     def _update_tqdm(arg, transform):
#         tqdm_bars[0].update(arg)

#     def _update_progress_bar(iter_num):
#         "Updates tqdm progress bar of a JAX scan or loop"
#         _ = jax.lax.cond(
#             iter_num == 0,
#             lambda _: host_callback.id_tap(_define_tqdm, None, result=iter_num
#                                            ),
#             lambda _: iter_num,
#             operand=None,
#         )

#         _ = jax.lax.cond(
#             # update tqdm every multiple of `print_rate` except at the end
#             (iter_num % print_rate == 0) &
#             (iter_num != num_samples - remainder),
#             lambda _: host_callback.id_tap(
#                 _update_tqdm, print_rate, result=iter_num),
#             lambda _: iter_num,
#             operand=None,
#         )

#         _ = jax.lax.cond(
#             # update tqdm by `remainder`
#             iter_num == num_samples - remainder,
#             lambda _: host_callback.id_tap(
#                 _update_tqdm, remainder, result=iter_num),
#             lambda _: iter_num,
#             operand=None,
#         )

#     def _close_tqdm(arg, transform):
#         tqdm_bars[0].close()

#     def close_tqdm(result, iter_num):
#         return jax.lax.cond(
#             iter_num == num_samples - 1,
#             lambda _: host_callback.id_tap(_close_tqdm, None, result=result),
#             lambda _: result,
#             operand=None,
#         )

#     def _progress_bar_scan(func):
#         """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
#         Note that `body_fun` must either be looping over `np.arange(num_samples)`,
#         or be looping over a tuple who's first element is `np.arange(num_samples)`
#         This means that `iter_num` is the current iteration number
#         """

#         def wrapper_progress_bar(carry, x):
#             if type(x) is tuple:
#                 iter_num, *_ = x
#             else:
#                 iter_num = x
#             _update_progress_bar(iter_num)
#             result = func(carry, x)
#             return close_tqdm(result, iter_num)

#         return wrapper_progress_bar

#     return _progress_bar_scan

# def _print_consumer(arg, transform):
#     iter_num, num_samples = arg
#     print(f"Iteration {iter_num:,} / {num_samples:,}")
#     print(datetime.datetime.fromtimestamp(time.time()).strftime('%c'))

# @jax.jit
# def progress_bar(arg, result):
#     """
#     Print progress of a scan/loop only if the iteration number is a multiple of
#     the print_rate

#     Usage: 'carry = progress_bar((iter_num + 1, num_samples, print_rate), carry)'
#     Pass in 'iter_num + 1' so that counting starts at 1 and ends at 'num_samples'

#     """
#     iter_num, num_samples, print_rate = arg
#     result = jax.lax.cond(
#         iter_num % print_rate == 0,
#         lambda _: host_callback.id_tap(_print_consumer,
#                                        (iter_num, num_samples),
#                                        result=result),
#         lambda _: result,
#         operand=None)
#     return result

# def progress_bar_scan(num_samples):

#     def _progress_bar_scan(func):
#         print_rate = int(num_samples / 10)

#         def wrapper_progress_bar(carry, iter_num):
#             iter_num = progress_bar((iter_num + 1, num_samples, print_rate),
#                                     iter_num)
#             return func(carry, iter_num)

#         return wrapper_progress_bar

#     return _progress_bar_scan

# import wandb

# wandb.login()
# wandb.init(project="comp646-project", entity="billqian")


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
        pi = distrax.Categorical(logits=actor_mean)

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

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(train_state, config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)

    network = ActorCritic(env.action_space().n,
                          activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state['params']

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    while not done:
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        # obs_batch = batchify(obs, env.agents, config["NUM_ACTORS"])
        # breakpoint()
        obs = {k: v.flatten() for k, v in obs.items()}

        pi_0, _ = network.apply(network_params, obs["agent_0"])
        pi_1, _ = network.apply(network_params, obs["agent_1"])

        actions = {
            "agent_0": pi_0.sample(seed=key_a0),
            "agent_1": pi_1.sample(seed=key_a1)
        }
        # env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
        # env_act = {k: v.flatten() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, actions)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


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
        network = ActorCritic(env.action_space().n,
                              activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)

        init_x = init_x.flatten()

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
        @scan_tqdm(int(config["NUM_UPDATES"]))
        # @progress_bar_scan(int(config["NUM_UPDATES"]))
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = batchify(last_obs, env.agents,
                                     config["NUM_ACTORS"])

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"],
                                     env.num_agents)

                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents,
                             config["NUM_ACTORS"]).squeeze(), action, value,
                    batchify(reward, env.agents,
                             config["NUM_ACTORS"]).squeeze(), log_prob,
                    obs_batch, info)
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state,
                                                    None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents,
                                      config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

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
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

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
                        entropy = pi.entropy().mean()

                        total_loss = (loss_actor +
                                      config["VF_COEF"] * value_loss -
                                      config["ENT_COEF"] * entropy)
                        return total_loss, (value_loss, loss_actor, entropy)

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

            # rewards = metric["returned_episode_returns"].mean(-1)
            # reward_mean = rewards.mean(0)
            # wandb.log({'return': reward_mean})

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )  # None for -2, config["NUM_UPDATES"] for -1
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None,
            config_path="config",
            config_name="ippo_ff_overcooked")
def main(config):
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
    execution_time = end_time - start_time
    print('Execution time: ', execution_time)

    print('** Saving Results **')
    filename = f'{config["ENV_NAME"]}_cramped_room_full'
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
    save_ckpt(train_state, config)
    ckpt = restore_ckpt()
    # print(ckpt['model']['params'])
    # print(ckpt['config'])
    state_seq = get_rollout(ckpt['model'], ckpt['config'])
    # state_seq = get_rollout(train_state, config)
    viz = OvercookedVisualizer()
    # agent_view_size is hardcoded as it determines the padding around the layout.
    viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")


def save_ckpt(train_state, config):
    ckpt_dir = '/Users/billqian/Documents/Graduate-Rice-University/' + \
        'Research/JaxMARL/baselines/IPPO/overcooked_ckpt'
    if os.path.exists(ckpt_dir):
        print('yes')
        shutil.rmtree(ckpt_dir)

    ckpt = {'model': train_state, 'config': config}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(ckpt_dir + '/orbax/single_save',
                            ckpt,
                            save_args=save_args)


def restore_ckpt():
    ckpt_dir = '/Users/billqian/Documents/Graduate-Rice-University/' + \
        'Research/JaxMARL/baselines/IPPO/overcooked_ckpt'

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(ckpt_dir + '/orbax/single_save')


if __name__ == "__main__":
    main()
