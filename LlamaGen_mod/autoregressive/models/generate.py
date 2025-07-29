# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
from builtins import print
from pyexpat.errors import codes
from tkinter import N
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits, _ = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits, _ = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def prefill_continous_code(model, cond_idx: torch.Tensor, mem: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float):
    if cfg_scale > 1.0:
        # cond_idx.shape = (64)
        # mem.shape = (32, 1, 16, 16)
        # input_pos.shape = (1)
        #print("cond_idx.shape:", cond_idx.shape, "mem.shape:", mem.shape)
        # print("cond_idx:", cond_idx)
        # print("cond_idx.shape:", cond_idx.shape)
        logits = model(codes=None, cond_idx=cond_idx, mem=mem, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits = model(codes=None, cond_idx=cond_idx, mem=mem, input_pos=input_pos)
        
    return logits   


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits, _ = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)

def decode_one_code(model, x: torch.Tensor, mem: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        mem_combined = torch.cat([mem, mem]) if mem is not None else None
        logits = model(codes=x_combined, cond_idx=None, mem=mem_combined, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits = model(codes=x, cond_idx=None, mem=mem, input_pos=input_pos)
    return logits

def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
                # input_pos.shape = (1)
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos = input_pos + 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    
    return new_tokens, new_probs


def decode_given_m_tokens(
    model, 
    given_tokens: torch.Tensor,  
    input_pos: torch.Tensor, 
    num_new_tokens: int,
    cfg_scale: float, 
    cfg_interval: int, 
    **sampling_kwargs):
    #print("given_tokens.shape:", given_tokens.shape)
    given_tokens_num = given_tokens.shape[1]
    #print("num of tokens given:", given_tokens_num)
    new_tokens = []
    cfg_flag = True
    cur_token = given_tokens[:, :1]
    #print("given_tokens.shape:", given_tokens.shape)# given_tokens.shape = (4, 255)
    #print("num_new_tokens:", num_new_tokens)# num_new_tokens = 255
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            if cfg_scale > -1 and i > cfg_interval:
                cfg_flag = False
            '''
            i = 0, given gt_token_1, decode gen_token_2
            '''
            next_token, _ = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos = input_pos + 1
            if i < given_tokens_num - 1:
                next_token = given_tokens[:, i+1:i+2]
            new_tokens.append(next_token.clone())
            cur_token = next_token.view(-1, 1)
    return new_tokens            


def decode_n_codes(
    model, cur_code: torch.Tensor, mem: torch.Tensor, input_pos: torch.Tensor, num_new_codes: int, 
    cfg_scale: float, cfg_interval: int):
    new_codes = []
    cfg_flag = True
    # print("num_new_codes:", num_new_codes)
    # print("mem.shape:", mem.shape if mem is not None else None)
    # print("input_pos:", input_pos)
    for i in range(num_new_codes):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            # cur_code.shape = (32, 1, 16)
            # mem.shape = (32, 255, 8)
            # input_pos.shape = (1)
            # cfg_flag = True
            # cfg_scale = 2.0
            next_code = decode_one_code(
                model, cur_code, mem[:, i:i+1, :] if mem is not None else None,
                input_pos, cfg_scale, cfg_flag)
            input_pos = input_pos + 1
            new_codes.append(next_code.clone())
            cur_code = next_code # todo: change shape
    return new_codes

def decode_n_codes_debug(
    model, cur_code, gt_codes: torch.Tensor, mem: torch.Tensor, input_pos: torch.Tensor, num_new_codes: int,
    cfg_scale: float, cfg_interval: int):
    codes_gt_context = []
    codes_gen_context = []
    cfg_flag = True
    for i in range(num_new_codes):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False

            next_code_gt_context = decode_one_code(
                model, gt_codes[:, i:i+1, :], mem[:, i:i+1, :] if mem is not None else None,
                input_pos, cfg_scale, cfg_flag
            )
            codes_gt_context.append(next_code_gt_context.clone())

            next_code_gen_context = decode_one_code(
                model, cur_code, mem[:, i:i+1, :] if mem is not None else None,
                input_pos, cfg_scale, cfg_flag
            )
            codes_gen_context.append(next_code_gen_context.clone())
            cur_code = next_code_gen_context

            input_pos = input_pos + 1            

    #codes_gen_context = torch.zeros_like(codes_gt_context)
    return codes_gt_context, codes_gen_context

# def decode_n_codes_debug(
#     model, cur_code: torch.Tensor, mem: torch.Tensor, input_pos: torch.Tensor, num_new_codes: int,
#     cfg_scale: float, cfg_interval: int):
#     new_codes = []
#     cfg_flag = True
#     for i in range(num_new_codes):
#         with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
#             if cfg_interval > -1 and i > cfg_interval:
#                 cfg_flag = False
#             next_code = decode_one_code(
#                 model, cur_code, mem[:, i:i+1, :] if mem is not None else None,
#                 input_pos, cfg_scale, cfg_flag
#             )
#             input_pos = input_pos + 1
#             new_codes.append(next_code.clone())
#             cur_code = next_code # todo: change shape
#     return new_codes

@torch.no_grad()
def generate(model, cond, max_new_tokens, emb_masks=None, 
            cfg_scale=1.0, cfg_interval=-1, **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    #print("type(input_pos):", type(input_pos), "input_pos:", input_pos)
    generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)
    
    model.del_caches()# important!

    return seq[:, T:]


@torch.no_grad()
def generate_with_given_tokens(model, cond, given_tokens, max_new_tokens, 
                                emb_masks=None, cfg_scale=1.0, 
                                cfg_interval=-1, **sampling_kwargs):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")

    T_new = T + max_new_tokens
    max_seq_length = T_new
    max_batch_size = cond.shape[0]

    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.tok_embeddings.weight.dtype)
    
    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
    
    # create an empty tensor of the expected final shape and fill in the current tokens
    seq = torch.empty((max_batch_size, T_new), dtype=torch.int, device=device)

    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)
    seq[:, T:T+1] = given_tokens[:, :1] if given_tokens is not None else next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    #print("type(input_pos):", type(input_pos), "input_pos:", input_pos), Tensor, 1
    #generated_tokens, _ = decode_n_tokens(model, next_token, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    generated_tokens = decode_given_m_tokens(model, given_tokens, input_pos, max_new_tokens-1, cfg_scale, cfg_interval, **sampling_kwargs)
    # print("len(generated_tokens):", len(generated_tokens))
    # print("max_new_tokens:", max_new_tokens-1)
    assert len(generated_tokens) == max_new_tokens-1

    # print("seq.shape:", seq.shape)
    # print("seq[:, T:].shape:", seq[:, T:].shape)
    # print("seq[:, T:T+1].shape:", seq[:, T:T+1].shape)
    # print("seq[:, T+1:].shape:", seq[:, T+1:].shape)
    # print("len(generated_tokens):", len(generated_tokens))
    # print("generated_tokens.shape:", torch.cat(generated_tokens, dim=1).shape)
    

    seq[:, T+1:] = torch.cat(generated_tokens, dim=1)

    model.del_caches()# important!

    return seq[:, T:]

def generate_debug(model, cond, gt_codes, mem, max_new_codes, emb_masks=None,
                   cfg_scale=1.0, cfg_interval=-1):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
            mem_combined = torch.cat([mem, mem]) if mem is not None else None
        else:
            cond_combined = cond
            mem_combined = mem
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
            mem_combined = torch.cat([mem, mem]) if mem is not None else None
        else:
            cond_combined = cond
            mem_combined = mem
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")
    
    T_new = T + max_new_codes
    max_seq_length = T_new
    max_batch_size = cond.shape[0]
    
    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.input_proj.weight.dtype)

    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
        
    seq_gt_context = torch.empty((max_batch_size, T_new, model.fvae.kl.embed_dim), dtype=torch.float32, device=device)
    seq_gen_context = torch.empty((max_batch_size, T_new, model.fvae.kl.embed_dim), dtype=torch.float32, device=device)

    input_pos = torch.arange(0, T, device=device)
    
    # cfg_scale == 1
    # cond_combined.shape = (b)
    # mem_combined.shape = (b, 256, 8)
    next_code = prefill_continous_code(model=model, cond_idx=cond_combined.long(), 
                                       mem=mem_combined[:, :1, :] if mem_combined is not None else None,
                                       input_pos=input_pos, cfg_scale=cfg_scale)
    seq_gt_context[:, T:T+1, :] = gt_codes[:, :1, :]
    seq_gen_context[:, T:T+1, :] = next_code

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    codes_gt_context, code_gen_context = decode_n_codes_debug(
        model, next_code, gt_codes, mem[:, 1:, :] if mem is not None else None,
        input_pos, max_new_codes-1, cfg_scale, cfg_interval
    )

    seq_gt_context[:, T+1:, :] = torch.cat(codes_gt_context, dim=1)
    seq_gen_context[:, T+1:, :] = torch.cat(code_gen_context, dim=1)
    #seq_gt_context[:, T+1:, :] = torch.zeros_like(seq_gen_context[:, T+1:, :]) # todo: change to zeros, for debug


    model.del_caches()
    
    return seq_gt_context[:, T:], seq_gen_context[:, T:]

def generate_continous_code(model, cond, mem, max_new_codes, emb_masks=None, cfg_scale=1.0, cfg_interval=-1):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
            mem_combined = torch.cat([mem, mem]) if mem is not None else None
        else:
            cond_combined = cond
            mem_combined = mem
        T = 1
    elif model.model_type == 't2i':
        if cfg_scale > 1.0:
            cond_null = torch.zeros_like(cond) + model.cls_embedding.uncond_embedding
            cond_combined = torch.cat([cond, cond_null])
            mem_combined = torch.cat([mem, mem]) if mem is not None else None
        else:
            cond_combined = cond
            mem_combined = mem
        T = cond.shape[1]      
    else:
        raise Exception("please check model type")
    
    T_new = T + max_new_codes
    max_seq_length = T_new
    max_batch_size = cond.shape[0]
    
    device = cond.device
    with torch.device(device):
        max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size
        model.setup_caches(max_batch_size=max_batch_size_cfg, max_seq_length=max_seq_length, dtype=model.input_proj.weight.dtype)

    if emb_masks is not None:
        assert emb_masks.shape[0] == max_batch_size
        assert emb_masks.shape[-1] == T
        if cfg_scale > 1.0:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * torch.cat([emb_masks, emb_masks]).unsqueeze(1)
        else:
            model.causal_mask[:, :, :T] = model.causal_mask[:, :, :T] * emb_masks.unsqueeze(1)

        eye_matrix = torch.eye(model.causal_mask.size(1), model.causal_mask.size(2), device=device)
        model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix
        
    seq = torch.empty((max_batch_size, T_new, model.fvae.kl.embed_dim), dtype=torch.float32, device=device)
    
    # print("\n")
    # print("max_batch_size:", max_batch_size)
    # print("seq.shape:", seq.shape)
    
    input_pos = torch.arange(0, T, device=device)
    
    # cfg_scale == 1
    # cond_combined.shape = (b)
    # mem_combined.shape = (b, 256, 8)
    next_code = prefill_continous_code(model=model, cond_idx=cond_combined, 
                                       mem=mem_combined[:, :1, :] if mem_combined is not None else None,
                                       input_pos=input_pos, cfg_scale=cfg_scale)
    # next_code.shape = (b, 1, 16)
    seq[:, T:T+1, :] = next_code

    # print("cfg_scale:", cfg_scale)
    # print("next_code.shape:", next_code.shape)
    # print("seq.shape:", seq.shape)
    # print("\n")
    
    
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    print("input_pos:", input_pos)
    
    #查看mem是否要[:, 1:, :]
    #确认decode_n_codes和decode_n_codes_debug的传参
    generated_codes = decode_n_codes(model, next_code, mem[:, 1:, :] if mem is not None else None,
                                     input_pos, max_new_codes-1, cfg_scale, cfg_interval)
    seq[:, T+1:, :] = torch.cat(generated_codes, dim=1)
    # print("seq[:, T+1:, :].shape:", seq[:, T+1:, :].shape)
    # print("seq[:, T:].shape:", seq[:, T:].shape)
    
    model.del_caches()
    
    return seq[:, T:]
    



def generate_code_wo_kvcache(model, cond, gt_codes, mem, seq_len=256, emb_masks=None, cfg_scale=1.0, cfg_interval=-1):
    if model.model_type == 'c2i':
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * model.num_classes
            cond_combined = torch.cat([cond, cond_null])
            mem_combined = torch.cat([mem, mem]) if mem is not None else None
        else:
            cond_combined = cond
            mem_combined = mem
    elif model.model_type == 't2i':
        raise NotImplementedError("generate_code_wo_kvcache is not implemented for t2i model type") 
    else:
        raise Exception("please check model type")
    
    max_batch_size = cond.shape[0]
    
    device = cond.device

    if emb_masks is not None:
        raise NotImplementedError("emb_masks is not supported in generate_code_wo_kvcache")

    gen_ar = torch.zeros((max_batch_size, seq_len, model.fvae.kl.embed_dim), dtype=torch.float32, device=device)
    gen_ar = gen_ar.to(dtype=next(model.parameters()).dtype)

    gt_context = torch.zeros_like(gen_ar) #作为context的ground truth codes
    gt_context = gt_context.to(dtype=next(model.parameters()).dtype)
    gen_gt = torch.zeros_like(gen_ar) #以ground truth codes作为context的生成codes
    gen_gt = gen_gt.to(dtype=next(model.parameters()).dtype)

    # print("seq.shape:", seq.shape)
    # 如果cfg_scale > 1.0, seq.shape == logits.shape == (b*2, 256, 16)
    # 否则seq.shape == logits.shape == (b, 256, 16)
    # print("cond_combined.shape:", cond_combined.shape)
    # print("mem_combined.shape:", mem_combined.shape if mem_combined is not None else None)
    

    # ------------------ suitable for different cfg_scale ----------------------- #
    # if cfg_scale > 1.0:
    #     logits_combined = model(codes=torch.cat([gen_ar, gen_ar]), cond_idx=cond_combined, mem=mem_combined, input_pos=torch.arange(seq_len))
    #     cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
    #     gen_ar_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    # else:
    #     gen_ar_logits = model(codes=gen_ar, cond_idx=cond_combined, mem=mem_combined, input_pos=torch.arange(seq_len))

    # assert gen_ar_logits.shape == gen_ar.shape

    # gen_ar[:, :1, :] = gen_ar_logits[:, :1, :]  # Fill the first token with logits
    # gen_gt[:, :1, :] = gen_ar_logits[:, :1, :]  # Fill the first token with logits
    
    # gt_context[:, :1, :] = gt_codes[:, :1, :]  # Fill the first token with gt_codes

    # for i in range(1, seq_len):
    #     gen_ar_logits = model(codes=gen_ar, cond_idx=cond, mem=mem, input_pos=torch.arange(seq_len))
    #     gen_ar[:, i:i+1, :] = gen_ar_logits[:, i:i+1, :]

    #     gen_gt_logits = model(codes=gt_context, cond_idx=cond, mem=mem, input_pos=torch.arange(seq_len))
    #     gt_context[:, i:i+1, :] = gt_codes[:, i:i+1, :]
    #     gen_gt[:, i:i+1, :] = gen_gt_logits[:, i:i+1, :]
    # --------------------------------------------------------------------------- #

    # -------------- without consideration about cfg_scale ---------------- #
    for i in range(seq_len):
        gen_ar_logits = model(gen_ar, cond_idx=cond, mem=mem, input_pos=torch.arange(seq_len))
        gen_ar[:, i:i+1, :] = gen_ar_logits[:, i:i+1, :]

        gen_gt_logits = model(gt_context, cond_idx=cond, mem=mem, input_pos=torch.arange(seq_len))
        gt_context[:, i:i+1, :] = gt_codes[:, i:i+1, :]
        
        gen_gt[:, i:i+1, :] = gen_gt_logits[:, i:i+1, :]
    # --------------------------------------------------------------------- #

    #seq_gt_context = model(codes=gt_codes, cond_idx=cond, mem=mem, input_pos=torch.arange(seq_len))

    return gen_ar, gen_gt

    


