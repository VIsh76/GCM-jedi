from torch import nn
import torch


embed_dim = 4
num_heads = 1

x = [
  [1, 0, 1, 0], # Seq 1
  [0, 2, 0, 2], # Seq 2
  [1, 1, 1, 1]  # Seq 3
 ]
x = torch.tensor(x, dtype=torch.float32)

w_key = [
  [0, 0, 1, 1],
  [1, 1, 0, 1],
  [0, 1, 0, 1],
  [1, 1, 0, 1]
]
w_query = [
  [1, 0, 1, 1],
  [1, 0, 0, 1],
  [0, 0, 1, 1],
  [0, 1, 1, 1]
]
w_value = [
  [0, 2, 0, 1],
  [0, 3, 0, 1],
  [1, 0, 3, 1],
  [1, 1, 0, 1]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)


# @ in the last dimensions
keys = (x @ w_key).unsqueeze(0)     # to batch mode
querys = (x @ w_query).unsqueeze(0)
values = (x @ w_value).unsqueeze(0)

# key and values are the same size (often they are the same tensor)


multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
attn_output, attn_output_weights = multihead_attn(querys, keys, values)
for p in multihead_attn.parameters():
    print(p, p.size())

print(querys.size())
print(x.size())
print(attn_output.size())
print(attn_output_weights.size())
