import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MiniTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, num_layers=3, ff_dim=512, max_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, ff_dim)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.reasoning_head = nn.Linear(d_model, vocab_size)
        self.answer_head = nn.Linear(d_model, vocab_size)
        self.tool_trigger_head = nn.Linear(d_model, 2)  # binary classification
        self.d_model = d_model

    def forward(self, input_ids, reasoning_labels=None, answer_labels=None):
        # (batch_size, seq_len) → (seq_len, batch_size)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # transformer expects (seq_len, batch, d_model)

        # Use last token as "query" into the sequence (hacky causal modeling)
        memory = x  # No encoder yet; just use embedding output as memory
        tgt = x.clone()

        decoded = self.transformer(tgt, memory)
        decoded = decoded.transpose(0, 1)  # back to (batch, seq_len, d_model)

        # Predict sequences
        reasoning_logits = self.reasoning_head(decoded)
        answer_logits = self.answer_head(decoded)

        # Tool trigger → use CLS token (first token)
        trigger_logits = self.tool_trigger_head(decoded[:, 0])

        return {
            'reasoning_logits': reasoning_logits,
            'answer_logits': answer_logits,
            'trigger_logits': trigger_logits
        }
