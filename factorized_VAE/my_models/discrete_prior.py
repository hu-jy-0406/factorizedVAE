import torch
import torch.nn as nn
from factorized_VAE.my_models.pos_emb import SinusoidalPositionalEmbedding
from factorized_VAE.utils import generate_causal_mask

    
class DiscretePrior(nn.Module):
    """
    Autoregressive Transformer using nn.TransformerDecoder.
    Predict token i given [CLS] and tokens < i.
    """
    def __init__(self, vocab_size, seq_len=64, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Embeddings + sinusoidal positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=seq_len)

        # TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        input: <s>, x1, x2, ...,xL-1
        output: x1',x2',x3',...,xL'
        """
        B, L = x.shape
        # Prepend <s> token (assume <s> ID = 0)
        start_token = torch.zeros((B, 1), dtype=torch.long, device=x.device)
        x = torch.cat([start_token, x], dim=1)  # => (B, L)
        x = x[:, :L]

        # Token embedding => shape (B, L, d_model)
        embeds = self.token_emb(x)

        # Add sinusoidal position embeddings => shape (B, L, d_model)
        h = self.pos_emb(embeds)

        # Create causal mask => shape (L, L)
        causal_mask = generate_causal_mask(L, x.device)

        # Dummy memory => shape (B, 1, d_model)
        memory = torch.zeros(B, 1, self.d_model, device=x.device)

        out = self.decoder(tgt=h, memory=memory, tgt_mask=causal_mask)
        out = self.ln(out)  # => (B, L, d_model)
        logits = self.head(out)  # => (B, L, vocab_size)

        return logits

    @torch.no_grad()
    def get_next_token(self, x):
        B, L = x.shape
        memory = torch.zeros(B, 1, self.d_model, device=x.device)  # Dummy memory for decoder

        embs = self.token_emb(x)  # shape (B, L, d_model)
        h = self.pos_emb(embs)  # shape (B, L, d_model)
        
        out = self.decoder(tgt=h, memory=memory, tgt_mask=None)
        out = self.ln(out)  # shape (B, L, d_model)
        logits = self.head(out)  # shape (B, L, vocab_size)
        last_logits = logits[:, -1, :]
        p = last_logits.softmax(dim=-1)  # Apply softmax to get probabilities
        #sample from the distribution p
        next_token = torch.multinomial(p, num_samples=1)
        return next_token  # shape (B, 1)


    @torch.no_grad()
    def generate(self, num_samples):
        device = next(self.parameters()).device

        token_list = []
        start_token = torch.zeros((num_samples,1)).long().to(device) # Start with <s> token
        token_list.append(start_token)  # Append <s> token ID (assumed to be 0)

        for i in range(self.seq_len):
            tokens = torch.cat(token_list, dim=1).to(device)  # shape (B, i+1)
            next_token = self.get_next_token(tokens)
            token_list.append(next_token) # Update the next token in the sequence

        tokens = torch.cat(token_list, dim=1)  # shape (B, seq_len+1)
        tokens = tokens[:,1:]

        return tokens