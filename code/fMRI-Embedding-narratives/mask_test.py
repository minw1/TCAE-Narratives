import torch
import torch.nn as nn

# Define input shapes
tgt_seq_len, memory_seq_len, batch_size, d_model = 5, 10, 2, 512

# Target and memory
tgt = torch.rand(tgt_seq_len, batch_size, d_model)  # (tgt_seq_len, batch_size, d_model)
memory = torch.rand(memory_seq_len, batch_size, d_model)  # (memory_seq_len, batch_size, d_model)

# Memory mask (for cross-attention)
memory_mask = torch.tensor([
    [0, 0, -float('inf'), -float('inf'), -float('inf')],
    [0, 0, 0, -float('inf'), -float('inf')],
    [0, 0, 0, 0, -float('inf')],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])  # Shape: (tgt_seq_len, memory_seq_len)

# Transformer Decoder
decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

# Pass inputs and mask to the decoder
output = decoder(
    tgt=tgt,
    memory=memory,
    memory_mask=memory_mask
)

print(output.shape)  # Output: (tgt_seq_len, batch_size, d_model)
