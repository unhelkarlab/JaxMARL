import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def create_and_prepare_model(model_name):
    # If using A100 (e.g. Colab Pro) you could use torch.bfloat16 instead.
    #compute_dtype = torch.bfloat16 # only on A100.
    compute_dtype = torch.float16

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=False,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        #attn_implementation = "flash_attention_2", # only on A100.
        device_map={"": 0})

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


#model, tokenizer = create_and_prepare_model("meta-llama/Llama-2-7b-chat-hf")
model, tokenizer = create_and_prepare_model(
    "mistralai/Mistral-7B-Instruct-v0.2")
model.eval()
print(model)
print(f"Vocabulary size: {tokenizer.vocab_size}")

prompt_text = """A famous person once said that life is difficult because everyone is looking for"""
# Show the input text prompt.
print(f"Input Text: {prompt_text}\n")

# Encode the text prompt into a tensor using the tokenizer.
prompt = tokenizer.encode(prompt_text, return_tensors='pt')

# Forward the input text prompt through the LM model.
output = model.forward(prompt)

# Get the outputs of the LM model.
print(f"Decoded Output: {tokenizer.decode(output.logits.argmax(-1)[0])}")
