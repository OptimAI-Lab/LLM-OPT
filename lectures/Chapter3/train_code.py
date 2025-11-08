import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect

from loguru import logger

from datasets import load_dataset

from transformers import get_constant_schedule_with_warmup
from transformers import AutoTokenizer

from torch.utils.data import IterableDataset, get_worker_info

###
# Pytorch implementation of causal self-attention (multi-head)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # ! n_embd = d_model, c_attn acts as all three W_q, W_k, W_v at once and all have output dim n_embd.
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
    
    # Pytorch implementation of Layer Normalization

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
    
    
# Pytorch implementation of MLP Layer

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
# Pytorch implemenation of nanoGPT

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
 
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    
# Environment parameters:
device = f"cuda:1"
workers = 0
# Data parameters:
batch_size = 64
# Tokenization parameters:
max_length = 256  # sequence length L
# Optimizer parameters
lr = 1e-3
weight_decay = 0.0 
# LR scheduler parameters
warmup_steps = 1000
# Training parameters:
num_training_steps = 10000
total_batch_size = 256
grad_clipping = 0.0 

data = load_dataset(
            "allenai/c4", "en", split="train", streaming=True
        )
val_data = load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )

class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        iter_data = iter(self.data)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        #input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        #attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])
        #return {"input_ids": input_ids, "attention_mask": attention_mask}
        
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        return input_ids    
    
    
# GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print("GPT-2 vocab size:", tokenizer.vocab_size)
 
# only used in eval ...
def preprocess_batched(batch):
    batch = tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return batch

dataset = PreprocessedIterableDataset(
                data, tokenizer, batch_size=batch_size, max_length=max_length
            )

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=None, num_workers=workers,
) 


# model parameters:
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False
block_size = 1024
vocab_size = 50304 #tokenizer.vocab_size
# ??? " vocab_size = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency"

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

n_total_params = sum(p.numel() for p in model.parameters())
trainable_params = [p for p in model.parameters() if p.requires_grad]


optimizer = torch.optim.Adam(
            trainable_params, lr=lr, weight_decay=weight_decay
        )

scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            last_epoch=-1,            
        )

global_step = 0
update_step = 0
tokens_seen = 0
tokens_seen_before = 0
world_size = 1

pad_idx = tokenizer.pad_token_id

gradient_accumulation = None

if  total_batch_size is not None:
    if  gradient_accumulation is None:
        assert (
            total_batch_size % world_size == 0
        ), "total_batch_size must be divisible by world_size"
        gradient_accumulation = total_batch_size // (
            batch_size * world_size
        )
        # logger.info(f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}")
        assert (
            gradient_accumulation > 0
        ), "gradient_accumulation must be greater than 0"

assert (
    gradient_accumulation * batch_size * world_size
    == total_batch_size
), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"


print(gradient_accumulation, "!!!")


# ##############################
# START of training loop
# ##############################

for batch_idx, batch in enumerate(dataloader):
    print(batch_idx)
    global_step += 1

    if update_step > num_training_steps:
        logger.info(
            f"Reached max number of update steps (f{num_training_steps}). Stopping training."
        )
        break


    input_ids = batch.to(device)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]   # shift left
    labels[:, -1] = pad_idx             # pad the last token
    labels[labels == pad_idx] = -100    # mask out padding
    labels = labels.to(device)
    tokens_seen += (input_ids != pad_idx).sum().item() * world_size

    print(labels)


    logits, loss = model(input_ids, targets=labels)
    scaled_loss = loss / gradient_accumulation
    scaled_loss.backward()

    if global_step % gradient_accumulation != 0:
        continue

    #######
    if grad_clipping != 0.0:
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clipping)
        
    #pbar.update(1)


    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
        
    update_step += 1
 
 
# ##############################
# END of training loop
# ##############################
logger.info("Training finished") 