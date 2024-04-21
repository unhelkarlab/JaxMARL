"""
This is the primary script for the Overcooked-Lang project. The ActorCritic 
class, and part of the get_rollout() and main() functions are borrowed from 
ippo_ff_overcooked.py script in the same directory. The rest of the code is 
authored by Bill Qian (zq4@rice.edu).
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
import hydra
from omegaconf import OmegaConf
import random
import os

import orbax.checkpoint

import imitation.data.types as types
import imitation.data.rollout as rollout
from imitation.data import serialize
from imitation.algorithms import bc
from imitation.util import logger as imit_logger

import gymnasium
import torch
from stable_baselines3.common import policies, torch_layers

from transformers import RobertaTokenizer

import wandb

# Maximum length of tokens encoded by RoBERTa
TOKENIZER_MAX_LENGTH = 16

# Different ways of saying that one is going to a location. Generated using
# Mistral-7B-Instruct-v0.2. The code for this is in the .ipynb notebook
llm_templates = [
    'I am going to the ', 'I am heading to the ', 'I\'m off to the ',
    'I\'m planning to visit the ', 'I\'m making my way to the ',
    'I\'ll be going to the '
]

# Path to the checkpoint that contains a policy for the agents in Overcooked
# trained using PPO
ckpt = '/Users/billqian/Documents/Graduate-Rice-University/' + \
        'Research/JaxMARL/baselines/IPPO/overcooked_ckpt'


class ActorCritic(nn.Module):
    """
    This is the underlying neural network for IPPO.
    """
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


def relabel_trajectory_helper(state_seq, obs_seq, action_seq, env_layout,
                              agent_idx, tokenizer):
    """
    Helper function for relabeling the trajectories with communication.
    """
    start_idx = 0
    end_idx = 0
    goal_seq = np.zeros((len(obs_seq), TOKENIZER_MAX_LENGTH), dtype=int)
    while end_idx < len(obs_seq):
        if end_idx == len(obs_seq) - 1:
            end_idx += 1
            continue

        if action_seq[end_idx] == 5:
            # print('interact')
            # print(end_idx)
            # Get the coords of the grid in front of the agent.
            grid_in_front = state_seq[end_idx].agent_pos + state_seq[
                end_idx].agent_dir
            grid_in_front_agent = grid_in_front[agent_idx]
            # print(grid_in_front_agent)
            # Convert the grid coords to the grid index.
            grid_idx = grid_in_front_agent[0] + grid_in_front_agent[1] * 5

            if grid_idx in env_layout['goal_idx']:
                # print('Goal')
                random_num = random.randrange(0, 6)
                text = llm_templates[random_num] + 'goal.'
                encoded_text = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=TOKENIZER_MAX_LENGTH,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='np')
                goal_seq[start_idx:end_idx + 1] = encoded_text['input_ids'][
                    0] * encoded_text['attention_mask'][0]
                start_idx = end_idx + 1
            elif grid_idx in env_layout['plate_pile_idx']:
                # print('Plate')
                random_num = random.randrange(0, 6)
                text = llm_templates[random_num] + 'plate pile.'
                encoded_text = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=TOKENIZER_MAX_LENGTH,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='np')
                goal_seq[start_idx:end_idx + 1] = encoded_text['input_ids'][
                    0] * encoded_text['attention_mask'][0]
                start_idx = end_idx + 1
            elif grid_idx in env_layout['onion_pile_idx']:
                # print('Onion')
                random_num = random.randrange(0, 6)
                text = llm_templates[random_num] + 'onion pile.'
                encoded_text = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=TOKENIZER_MAX_LENGTH,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='np')
                goal_seq[start_idx:end_idx + 1] = encoded_text['input_ids'][0]
                start_idx = end_idx + 1
            elif grid_idx in env_layout['pot_idx']:
                # print('Pot')
                random_num = random.randrange(0, 6)
                text = llm_templates[random_num] + 'pot.'
                encoded_text = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=TOKENIZER_MAX_LENGTH,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='np')
                goal_seq[start_idx:end_idx + 1] = encoded_text['input_ids'][
                    0] * encoded_text['attention_mask'][0]
                start_idx = end_idx + 1
            # else:
            # print('Other')

        end_idx += 1

    return goal_seq


def relabel_trajectory(state_seq, obs_seq_0, action_seq_0, obs_seq_1,
                       action_seq_1, env_layout, tokenizer):
    """
    Relabel the trajectories with communication and return the new
    observations.
    """
    goal_seq_0 = relabel_trajectory_helper(state_seq, obs_seq_0, action_seq_0,
                                           env_layout, 0, tokenizer)
    goal_seq_1 = relabel_trajectory_helper(state_seq, obs_seq_1, action_seq_1,
                                           env_layout, 1, tokenizer)
    obs_seq_0 = np.concatenate((obs_seq_0, goal_seq_0, goal_seq_1), axis=1)
    obs_seq_1 = np.concatenate((obs_seq_1, goal_seq_1, goal_seq_0), axis=1)

    return obs_seq_0, obs_seq_1


def get_rollout(train_state, config, num_traj=1, lang=False, tokenizer=None):
    """
    Unroll a trained policy to obtain a list of trajectories.
    """
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # env_params = env.default_params
    # env = LogWrapper(env)
    env_layout = config["ENV_KWARGS"]["layout"]

    network = ActorCritic(env.action_space().n,
                          activation=config["ACTIVATION"])
    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    init_x = jnp.zeros(env.observation_space().shape)
    init_x = init_x.flatten()

    network.init(key_a, init_x)
    network_params = train_state['params']

    all_trajs = []
    for i in range(num_traj):
        done = False

        obs, state = env.reset(key_r)
        state_seq = [state]

        obs_init = {k: v.flatten() for k, v in obs.items()}
        obs_seq_0 = [obs_init["agent_0"]]
        obs_seq_1 = [obs_init["agent_1"]]
        action_seq_0 = []
        action_seq_1 = []

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
            action_seq_0.append(actions["agent_0"])
            action_seq_1.append(actions["agent_1"])
            # env_act = unbatchify(action, env.agents, config["NUM_ENVS"],
            #                      env.num_agents)
            # env_act = {k: v.flatten() for k, v in env_act.items()}

            # STEP ENV
            obs, state, reward, done, info = env.step(key_s, state, actions)
            done = done["__all__"]

            obs_log = {k: v.flatten() for k, v in obs.items()}
            obs_0 = obs_log["agent_0"]
            obs_1 = obs_log["agent_1"]
            obs_seq_0.append(obs_0)
            obs_seq_1.append(obs_1)

            state_seq.append(state)

            # counter += 1
            # print(counter)

        if not lang:
            traj_0 = types.Trajectory(np.asarray(obs_seq_0),
                                      np.asarray(action_seq_0), None, True)
            traj_1 = types.Trajectory(np.asarray(obs_seq_1),
                                      np.asarray(action_seq_1), None, True)
            all_trajs.append(traj_0)
            all_trajs.append(traj_1)
        else:
            # Relabel trajectories.
            action_seq_0 = np.asarray(action_seq_0)
            action_seq_1 = np.asarray(action_seq_1)
            obs_seq_0, obs_seq_1 = relabel_trajectory(state_seq,
                                                      np.asarray(obs_seq_0),
                                                      action_seq_0,
                                                      np.asarray(obs_seq_1),
                                                      action_seq_1, env_layout,
                                                      tokenizer)

            traj_0 = types.Trajectory(obs_seq_0, action_seq_0, None, True)
            traj_1 = types.Trajectory(obs_seq_1, action_seq_1, None, True)
            all_trajs.append(traj_0)
            all_trajs.append(traj_1)

    # Save trajectories.
    if not lang:
        serialize.save('trajs/unlabeled_trajs', all_trajs)
    else:
        serialize.save('trajs/labeled_trajs', all_trajs)
    return all_trajs


def restore_ckpt():
    """
    Restore the checkpoint that contains a trained policy and configuration
    parameters.
    """
    ckpt_dir = ckpt
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(ckpt_dir + '/orbax/single_save')


class CustomBCLogger(bc.BCLogger):
    """
    Create a custom logger to log additional info and output all the info to 
    wandb.
    """

    def __init__(self, logger, test_transitions, bc_trainer, wandb_run=None):
        """
        Args:
        - bc_trainer: the behavior cloning agent that is being trained
        - wandb_run: a run object returned from wandb.init()
        """
        super().__init__(logger)
        self.test_transitions = test_transitions
        self.bc_trainer = bc_trainer
        self.wandb_run = wandb_run

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        num_samples_so_far: int,
        training_metrics,
        rollout_stats,
    ):
        self._logger.record("batch_size", batch_size)
        self._logger.record("bc/epoch", self._current_epoch)
        self._logger.record("bc/batch", batch_num)
        self._logger.record("bc/samples_so_far", num_samples_so_far)
        for k, v in training_metrics.__dict__.items():
            value = float(v) if v is not None else None
            name = f"bc/{k}"
            if name == 'bc/loss':
                name = 'bc/training_loss'
            self._logger.record(name, value)
            if self.wandb_run is not None:
                self.wandb_run.log({name: value})

        # Compute test loss
        with torch.no_grad():
            training_metrics = self.bc_trainer.loss_calculator(
                self.bc_trainer.policy, self.test_transitions.obs,
                self.test_transitions.acts)
            self._logger.record('bc/test_loss', training_metrics.loss.item())
            if self.wandb_run is not None:
                self.wandb_run.log(
                    {'bc/test_loss': training_metrics.loss.item()})

        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0


class FeedForward64Policy(policies.ActorCriticPolicy):
    """
    A feed forward policy network with two hidden layers of 64 units.
    """

    def __init__(self, *args, **kwargs):
        """
        Builds FeedForward64Policy; arguments passed to `ActorCriticPolicy`.
        """
        super().__init__(*args, **kwargs, net_arch=[64, 64])


def train_bc_agent(obs_space,
                   action_space,
                   train_transitions,
                   test_transitions,
                   rng,
                   lang,
                   n_epochs=20):
    """
    Train a behavior cloning agent.
    """
    # Set up a behavior cloning agent.
    if lang:
        extractor = (torch_layers.CombinedExtractor if isinstance(
            obs_space, gymnasium.spaces.Dict) else
                     torch_layers.FlattenExtractor)
        policy = FeedForward64Policy(
            observation_space=obs_space,
            action_space=action_space,
            # Set lr_schedule to max value to force error if policy.optimizer
            # is used by mistake (should use self.optimizer instead).
            lr_schedule=lambda _: torch.finfo(torch.float32).max,
            features_extractor_class=extractor,
        )
        bc_trainer = bc.BC(observation_space=obs_space,
                           action_space=action_space,
                           demonstrations=train_transitions,
                           rng=rng,
                           policy=policy)
    else:
        bc_trainer = bc.BC(observation_space=obs_space,
                           action_space=action_space,
                           demonstrations=train_transitions,
                           rng=rng)
    # Login and initialize wandb
    wandb.login()
    run = wandb.init(project='646-project', entity='billqian')
    # Use a custom logger for the behavior cloning agent.
    curr_dir = os.getcwd()
    logging_dir = os.path.join(curr_dir, 'results/')
    logger = imit_logger.configure(logging_dir, ['stdout', 'csv'])
    custom_logger = CustomBCLogger(logger,
                                   test_transitions,
                                   bc_trainer,
                                   wandb_run=run)
    bc_trainer._bc_logger = custom_logger
    # Train the behavior cloning agent.
    bc_trainer.train(n_epochs=n_epochs)


@hydra.main(version_base=None,
            config_path="config",
            config_name="ippo_ff_overcooked")
def main(config):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    config = OmegaConf.to_container(config)
    config["ENV_KWARGS"]["layout"] = overcooked_layouts[config["ENV_KWARGS"]
                                                        ["layout"]]
    ckpt = restore_ckpt()
    lang = True
    # print(ckpt['model']['params'])
    # print(ckpt['config'])
    print('Collect trajectories...')
    # get_rollout(ckpt['model'],
    #             ckpt['config'],
    #             num_traj=50,
    #             lang=lang,
    #             tokenizer=tokenizer)

    # Load trajectories and flatten into transitions
    if lang:
        all_trajs = serialize.load('trajs/labeled_trajs')
    else:
        all_trajs = serialize.load('trajs/unlabeled_trajs')
    # Divide the trajectories into a training set and a test set
    all_trajs = list(all_trajs)
    print(len(all_trajs[0].acts))
    return
    random.Random(0).shuffle(all_trajs)
    train_frac = 0.8
    train_size = int(len(all_trajs) * train_frac)
    train_transitions = rollout.flatten_trajectories(all_trajs[:train_size])
    test_transitions = rollout.flatten_trajectories(all_trajs[train_size:])

    # Prepare to train a Behavior Cloning (BC) agent
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    rng = 0
    if lang:
        obs_space = gymnasium.spaces.Box(
            0, 50265, (env.num_obs + TOKENIZER_MAX_LENGTH * 2, ))
    else:
        obs_space = gymnasium.spaces.Box(0, 255, (env.num_obs, ))
    action_space = gymnasium.spaces.Discrete(env.num_actions, seed=rng)
    print('Train BC agent...')
    train_bc_agent(obs_space, action_space, train_transitions,
                   test_transitions, rng, lang)

    # viz = OvercookedVisualizer()
    # # agent_view_size is hardcoded as it determines the padding around the layout.
    # filename = f'{config["ENV_NAME"]}_cramped_room_full'
    # viz.animate(state_seq, agent_view_size=5, filename=f"{filename}.gif")


if __name__ == "__main__":
    main()