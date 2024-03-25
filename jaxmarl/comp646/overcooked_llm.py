"""
Short introduction to running the Overcooked environment and visualising it using random actions.
"""

import jax
from jaxmarl import make
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts, layout_grid_to_dict
import time

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

mps_device = torch.device('mps')

print('Load model and tokenizer')
model_dir = "./llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
model = LlamaForCausalLM.from_pretrained(
    model_dir,
    load_in_8bit=False,
    #  quantization_config=bnb_config,
    device_map='auto',
    offload_folder='offload',
    torch_dtype=torch.float16)
print(model.device)
model.eval()
tokenizer = LlamaTokenizer.from_pretrained(model_dir)


def generate_sentence(model,
                      tokenizer,
                      prompt_text,
                      max_length=30,
                      temperature=0.9):

    def gen_sentence_helper(model, tokenizer, encoded_prompt, max_length,
                            temperature):
        # Pass the prompt through the model.
        output = model.forward(encoded_prompt)

        # Get the scaled logits for the next token.
        next_token_scores_scaled = output.logits[0][-1].data.to(
            "cpu") / temperature

        # Get a probability distribution of the next token.
        norm_next_token_scores = next_token_scores_scaled.softmax(axis=0)

        # Sample the next token.
        next_token = norm_next_token_scores.multinomial(1).to(mps_device)

        # Add the token to the prompt.
        encoded_prompt = torch.cat(
            [encoded_prompt, next_token.unsqueeze(0)], dim=-1)

        # if (next_token == tokenizer.eos_token_id or
        if (next_token == 28723 or encoded_prompt.shape[1] >= max_length):
            return encoded_prompt

        return gen_sentence_helper(model, tokenizer, encoded_prompt,
                                   max_length, temperature)

    with torch.no_grad():
        # Your code goes here.

        # Encode the text prompt into a tensor using the tokenizer.
        prompt = tokenizer.encode(prompt_text,
                                  return_tensors='pt').to(mps_device)
        prompt_length = len(prompt[0])

        tokens = gen_sentence_helper(model, tokenizer, prompt,
                                     max_length + prompt_length,
                                     temperature)[0]

        return tokenizer.decode(tokens[prompt_length:]).strip()


# Your prompt goes here.
prompt_text = """Imagine an image of nature. What's in this image?
                 Begin your answer with "This image shows a". Limit your answer with 15 words."""

start = time.time()
print(generate_sentence(model, tokenizer, prompt_text, max_length=30))
end = time.time()
print('Time: ', start - end)
# print(generate_sentence(model, tokenizer, prompt_text, max_length=30))
# print(generate_sentence(model, tokenizer, prompt_text, max_length=30))
# print(generate_sentence(model, tokenizer, prompt_text, max_length=30))
# print(generate_sentence(model, tokenizer, prompt_text, max_length=30))

# print('Prepare model for inference')
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# print('Run the pipeline')
# sequences = pipeline(
#     'How are you doing?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=50,
# )

# for seq in sequences:
#     print(f"{seq['generated_text']}")

initial_prompt = """Overcooked is a cooperative cooking simulation video game 
where players control a team of chefs working in kitchens to prepare and cook 
orders under a time limit.
You will use the informatin provided below to determine the best action for both
agent 0 and agent 1. 
"""
# def feed_initial_prompt():

# # Parameters + random keys
# num_episodes = 5
# max_steps = 128
# key = jax.random.PRNGKey(0)
# key, key_r, key_a = jax.random.split(key, 3)

# # Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
# layout = overcooked_layouts["cramped_room"]

# # Instantiate environment
# env = make('overcooked', layout=layout, max_steps=max_steps)

# obs, state = env.reset(key_r)
# print('list of agents in environment', env.agents)
# print('obs: ', obs)

# # Sample random actions
# key_a = jax.random.split(key_a, env.num_agents)
# actions = {
#     agent: env.action_space(agent).sample(key_a[i])
#     for i, agent in enumerate(env.agents)
# }
# print('example action dict', actions)

# for episode in range(num_episodes):
#     obs, state = env.reset(key_r)
#     episode_r_a0 = 0
#     episode_r_a1 = 0

#     for _ in range(max_steps):
#         # Iterate random keys and sample actions
#         key, key_s, key_a = jax.random.split(key, 3)
#         key_a = jax.random.split(key_a, env.num_agents)

#         actions = {
#             agent: env.action_space(agent).sample(key_a[i])
#             for i, agent in enumerate(env.agents)
#         }

#         # Step environment
#         obs, state, rewards, dones, infos = env.step(key_s, state, actions)
#         episode_r_a0 += rewards['agent_0']
#         episode_r_a1 += rewards['agent_1']

#         # if dones['__all__']:
#         #     break

#     print(episode_r_a0)
#     print(episode_r_a1)

# viz = OvercookedVisualizer()

# Render to screen
# for s in state_seq:
#     viz.render(env.agent_view_size, s, highlight=False)
#     time.sleep(0.25)

# # Or save an animation
# viz.animate(state_seq, agent_view_size=5, filename='animation.gif')
