# Need to call this before importing transformers.
# from llapsa.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()
from llapsa.train.train import train

if __name__ == "__main__":
    train()
