import os
import time
import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn= nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj= nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT= 1

        self.n_head= config.n_head
        self.n_embd= config.n_embd

    def forward(self, x):
        assert self.n_embd % self.n_head== 0
        B, T, C= x.size()
        qkv= self.c_attn(x)
        q, k, v= qkv.split(self.n_embd, dim=-1)
        k= k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q= q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v= v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y= F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y= y.transpose(1, 2).contiguous().view(B, T, C)
        y= self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc= nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu= nn.GELU()
        self.c_proj= nn.Linar(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT= 1

    def forward(self, x):
        x= self.c_fc(x)
        x= self.gelu(x)
        x= self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1= nn.LayerNorm(config.n_embd)
        self.attn= CausalSelfAttention(config)
        self.ln_2= nn.LayerNorm(config.n_embd)
        self.mlp= MLP(config)

    def forward(self, x):
        x= x + self.attn(self.lm_1(x))
        x= x + self.mlp(self.ln_2(x))

        return x


@dataclass
class GPTConfig:
    vocab_size: int=50257
    block_size: int=1024
    n_embd: int= 768
    n_head: int = 12
    n_layer: int= 12


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config= config
        self.transformers= nn.ModuleDict(dict(
            wte= nn.Embedding(config.n_embd),
            wpe= nn.Embedding(config.n_embd),
            h= nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd)
        ))
        self.lm_head= nn.Linear(config.n_embd, config.vocab_size)

        self.transformers.wte.weight= self.lm_head.weight

        self.apply(self._init_weights_)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std= 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean= 0.0, std= 0.02)


    def forward(self, idx, targets=None):
        B, T= idx.size()
        pos= torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb= self.transformers.wpe(pos)
        tok_emb= self.transformers.wte(idx)
        x= tok_emb + pos_emb
        for block in self.transformers.h:
            x= block(x)
        x= self.transformers.ln_f(x)
        logits= self.lm_head(x)
        loss= None

        if targets is not None:
            loss= F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ['gpt2', 'gpt2_medium', 'gpt2_large', 'gpt2_xl']
        print(f'loading model weights from pretrained:{model_type}')

        from transformers import GPT2LMHeadModel

        config_args= {
            'gpt2'       :   dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2_medium':   dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt_large'  :   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt_xl'     :   dict(n_layer=48, n_head=25, n_embd= 1600),
        } [model_type]

        config_args['vocab_size']= 50257
        config_args['block_size']= 1024

        config= GPTConfig(**config_args)
        model= GPT(config)
        sd= model.state_dict()
        sd_keys= sd.keys()
        sd_keys= [k for k in sd_keys if k.endswith('.attn_bias')]

        model_hf= GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf= model_hf.state_dict()
        sd_keys_hf= sd_hf.keys()
        sd_keys_hf= [k for k in sd_keys_hf if k.endswith('.attn_masked_bias')]
        sd_keys_hf= [k for k in sd_keys_hf if k.endswith('.attn_bias')]
        transposed= ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if any(k.startswith(w) for w in transposed):
                assert sd_keys_hf[k].shape[::-1]== sd_keys[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_keys_hf[k].shape== sd_keys[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    
    def configure_optimizer(self, weight_decay, learning_rate, device_type):
        param_dict= {pn:p for pn, p in self.named_parameters()}
        param_dict= {pn:p for pn, p in param_dict.items() if p.requires_grad}

        decay_params= [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params= [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups= [
            {'param' :   decay_params, 'weight_decay'    :  weight_decay},
            {'param' :   nodecay_params, 'weight_decay'  :  0.0}
        ]

        num_decay_params= sum(p.numel() for p in decay_params)
        num_nodecay_params= sum(p.numel() for p in nodecay_params)

        if master_process:
            print(f'num decay parameter tensor: {len(decay_params)} with {num_decay_params} parameters')
            print(f'num nodecay parameter tensor: {len(nodecay_params)} with {num_nodecay_params} parameters')

        fused_availaible= 'fused' if inspect.signature(torch.optim.AdamW).parameters
        use_fused= fused_availaible and device_type== 'cuda'

        if master_process:
            print(f'uing fused AdamW {use_fused}')

        optimizer= torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-6, fused=use_fused)

        return optimizer

    #---------------------------------------------------------------------------------------------------------------------

import tiktoken
import numpy as np

def load_tokens(filename):
    npt= np.load(filename)
    npt= npt.astype(np.int32)
    ptt= torch.tensor(npt, dtype=torch.long)

    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B= B
        self.T= T
        self.process_rank= process_rank
        num_processes= num_processes
        assert split in ['train', 'val']

        data_root= 'edu_fineweb10b'
        shards= os.listdir(data_root)
        shards= [s for s in shards if split in s]
        shards= sorted(shards)
        shards= [os.path.join(data_root, s) for s in shards]
        self.shards= shards
        self.reset()

    def reset(self):
        current_shard= 0
        self.tokens= load_tokens(self.shards[current_shard])
        self.current_position= self.B * self.T * self.process_rank

    def next_batch(self):
        B, T= self.B, self.T
        buf= self.tokens[self.current_position: self.current_position + B * T +1]
        x= buf[:-1].view(B, T)
        y= buf[1:].view(B, T)

        self.current_position+= B * T * self.num_processes

        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_shard= (self.current_shard + 1) % len(self.shards)
            self.tokens= load_tokens(self.current_shard)
            self.current_position= B * T * self.process_rank

        return x, y

#------------------------------------------------------------------------------------------------

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp= int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank= int(os.environ['RANK'])
    ddp_local_rank= int(os.environ['LOCAL_RANK'])
    ddp_world_size= int(os.environ['WORLD_SIZE'])
    device= f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process= ddp_rank== 0

else:
    ddp_rank= 0
    ddp_local_rank= 0
    ddp_world_size= 1
    master_process= True

    device= 'cpu'
    if torch.cuda.is_available():
        device= 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device= 'mps'

device_type= 'cuda' if device.startswith('cuda') else 'cpu'


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc= tiktoken.get_encoding('gpt2')

total_batch_size= 524288
B= 64
T= 1024
assert total_batch_size % B * T * ddp_world_size== 0
grad_accum_step= total_batch_size // B * T * ddp_world_size

if master_process:
    print(f'total batch size: {total_batch_size}')
    print(f'grad accum steps: {grad_accum_step}')

train_loader= DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader= DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

torch.set_float32_matmul_precision('high')

model= GPT(GPTConfig(vocab_size=50304))
model.to(device)

use_compile= False
if use_compile:
   model=  torch.compile(model)
if ddp:
    model= DDP(model, device_ids=[ddp_local_rank])

raw_model= model.module if ddp else model

max_lr= 0.0006
min_lr= 0.1 * max_lr
warmup_steps= 715
max_steps= 19073
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio= (it - warmup_steps) / (max_steps - warmup_steps)
    coeff= 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer= raw_model.configure_optimizer(weight_decay=0.1, learning_rate=0.0006, device_type=device_type)

log_dir= 'log'
os.makedirs(log_dir, exist_ok=True)
log_file= os.path.join(log_dir, f'log.txt')
with open(log_file, 'w') as f:
    pass


for step in range(max_steps):
    t0= time.time()
    last_step= (step== max_steps - 1)

    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum= 0.0
            val_loss_step= 20
            for _ in range(val_loss_step):
                x, y= val_loader.next_batch()
                x, y= x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss= model(x, y)
                loss= loss / val_loss_step
                val_loss_accum+= loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f'{step} val loss: {val_loss_accum.item():.4f}')
            with open(log_file, 'a') as f:
                f.write(f'step {step} val_loss: {val_loss_accum.item():.4f}\n')
            if step > 0 and (step % 5000== 0 or last_step):
                checkpoint_path= os.path.join(log_file, f'model_{step}.pt')
                checkpoint= {
                    'model'      :  raw_model.state_dict(),
                    'config'     :  raw_model.config,
                    'step'       :  step,
                    'val_loss'   :  val_loss_accum.item()   
                }
                torch.save(checkpoint, checkpoint_path)

    if ((step > 0 and step % 250== 0) or last_step) and (not use_compile):
        with torch.no_gra():
            model.eval()
            num_return_sequences= 4
            max_length= 32
            tokens= enc.encode("Hello, i'm a language model")
            tokens= torch.unsqueeze(0).repeat(num_return_sequences)
            tokens= torch.tensor(tokens, dtype=torch.long)
            xgen= tokens.to(device)
            sample_rng= torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)

            while xgen < range(max_length):
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                        logits= logits[:, -1, :]
                        probs= F.softmax(logits, dim=-1)
                        topk_probs, topk_indices= torch.topk(probs, 50, dim=-1)
                        ix= torch.multinomial(topk_probs, 1, generator=sample_rng)
                        xcol= torch.gather(topk_indices, -1, ix)
                        xgen= torch.cat((xgen, xcol), dim=1)

                for i in range(num_return_sequences):
                    tokens= xgen[i, :max_length].tolist()
                    decoder= enc.decode(tokens)
                    print(f'rank {ddp_rank}, sample {i}: {decoder}')


    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum=0.0
    for micro_step in range(grad_accum_step):
        x, y= train_loader.next_batch()
        x, y= x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync= (micro_step== grad_accum_step - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss= model(x, y)
        loss= loss / grad_accum_step
        loss_accum+= loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm= torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr= get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']= lr

    optimizer.step()

    t1= time.time()
    dt= t1 - t0
    tokens_processed= train_loader.B * train_loader.T * grad_accum_step * ddp_world_size
    tok_per_sec= tokens_processed / dt

    if master_process:
        print(f'step {step} | loss: {loss_accum.item():.4f} | lr:{lr:.4f} | norm:{norm:.4f} | dt:{dt*1000:.2f}ms | tok/sec:{tok_per_sec:.2f}')
        with open(log_file, 'a') as f:
            f.write(f'{step}, loss: {loss_accum.item():.4f}')

if ddp:
    destroy_process_group()


