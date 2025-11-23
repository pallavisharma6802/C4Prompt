from app.compressor import compress

s = (
    "Initial Training Phase for TSV\n\n"
    "The Trainable Steering Vector (TSV) is a single vector, v ∈R d , trained to detect hallucinations in Large Language Models (LLMs) without altering the model's core architecture. "
    "The TSV is added to the hidden states of the LLM at an intermediate layer, l, to guide the model's output. The training objective is to learn the optimal v that can distinguish between truthful and hallucinated data."
)

c, m = compress(s)
from app.compressor import count_tokens

print('--- ORIGINAL ---')
print(s)
print('Original tokens:', count_tokens(s))
print('\n--- COMPRESSED ---')
print(c)
print('Compressed tokens:', count_tokens(c))
print('\n--- META ---')
import json
print(json.dumps(m, indent=2))
