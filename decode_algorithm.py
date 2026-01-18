import torch
from dllm_model import BidirectionalDLLM, CausalDLLM, BlockDynamicCache, split_cache, merge_cache
from dataclasses import asdict, dataclass, field
from typing import List, Tuple, Optional, Union, Any
import torch.nn.functional as F
import torch.distributions as dists

@dataclass
class DecodeConfig:
    # window_type: 'static' or 'sliding'.
    # 'static' for traditional block wise decoding
    # 'sliding' for our proposed method
    window_type: str = 'sliding'

    generate_length: int = 512
    initial_window_length: int = 32
    block_size: int = 32
    temperature: float = 0.0
    # algo for decoding algorithms:
    # fixed: decode fixed number of masks in window.
    # threshold: decode masks with confidence >= threshold (and at least 1).
    # factor: see Fast-DLLM V1 paper.
    decode_algo: str = 'threshold'
    decode_param: Union[int, float] = 0.9
    # Maximum window length
    max_window_length: int = 1000000000
    cache_type: str = 'none'
    # for cache_type starts with 'prefix' or 'dual', the cache refreshes if more than `refresh_count` masks are decoded since last refreshment.
    # for cache_type = 'none' or causal_decode , this parameter is useless.
    refresh_count: int = 16
    # if True, return (ids, List[DebugInfo]), else return (ids, None)
    debug: bool = False
    # special ids, default by LLADA-8B-Instruct
    mask_id: int = 126336
    pad_id: int = 126081
    eos_id: int = 126081
    # For any usage
    kwargs: dict = field(default_factory=dict)


@dataclass
class OneStepDebugInfo:
    '''It describes information in one decoding step'''
    # Number of tokens fed into bottom model each seqeunce
    forward_length: int
    # Decode window range (i.e. [window_l, window_r]).
    window_range: Tuple[int, int]
    # Decode indexes
    decode_indices: List[torch.Tensor]
    # Decode confidence
    decode_confidence: List[torch.Tensor]
    # After this iteration, what are the token ids. Sahpe (batch_size, total_length)
    token_after_decode: Optional[torch.Tensor] = None
    # In this iteration, what are the token prediction result. Shape (batch_size, window_range).
    token_generated: Optional[torch.Tensor] = None
    # In this iteration, what is the predicted token's confidence. Shape (batch_size, window_range).
    confidence_generated: Optional[torch.Tensor] = None
    # Whether to use cache (if False, refresh all the caches)
    pass_cache: bool = False
    # For any usage
    reserved_field: Any = None

torch.serialization.add_safe_globals([DecodeConfig, OneStepDebugInfo])

#####################
#### From NBDiff ####
#####################

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0

#############
#### END ####
#############

def get_decode_num(confidence: torch.Tensor, candidates: torch.Tensor, algo: str, algo_param: Union[int, float]) -> torch.Tensor:
    '''Calculate how many tokens will be decoded each sequence.
    - candidates: dtype=bool, True represents the token can be decode and vice versa.'''
    if algo.endswith('fixed'):
        return torch.min(torch.sum(candidates, dim=1), torch.tensor(algo_param)).long()
    elif algo.endswith('threshold'):
        return torch.max(torch.sum(candidates & (confidence >= algo_param), dim=1), torch.any(candidates, dim=1)).long()
    else:
        raise ValueError(f"Unsupported decoding algorithm {algo}")

def get_decode_conf_and_indices(confidence: torch.Tensor, candidates: torch.Tensor, num_decode: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''For each sequence, find decoding positions' confidence and indices'''
    confidence = torch.where(candidates, confidence, -torch.inf)
    conf, ind = [], []
    for i in range(confidence.shape[0]):
        topk = torch.topk(confidence[i], num_decode[i].item())
        conf.append(topk[0])
        ind.append(topk[1])
    return conf, ind

def floor_divisible(a: int, b: int) -> int:
    '''find interger x where x<=a and x%b=0 '''
    return a // b * b

def ceil_divisible(a: int, b: int) -> int:
    '''find interger x where x>=a and x%b=0 '''
    return (a + b - 1) // b * b

def window_bidirectional_decode(
    model: BidirectionalDLLM,
    input_ids: torch.Tensor,
    config: DecodeConfig
) -> Tuple[torch.Tensor, Optional[List[OneStepDebugInfo]]]:
    ''' This function runs bidirectional diffusion LLM generation in sliding window / static window (block) based decoding strategy.'''

    # kwargs={'expansion_max': 2, 'expansion_step': 1, 'expansion_threshold': 0.45}

    generate_length = config.generate_length
    window_size = config.initial_window_length
    block_size = config.block_size
    max_window_length = config.max_window_length

    device = input_ids.device
    batch_size, initial_length = input_ids.shape
    full_length = initial_length + generate_length
    debug_info = [] if config.debug else None

    # The window is [window_left, window_right) for all batches.
    if config.window_type == 'sliding':
        window_left, window_right = initial_length, initial_length + window_size
        num_mask = [window_size for _ in range(batch_size)]
    elif config.window_type == 'static':
        window_left, window_right = initial_length, initial_length + block_size
    else:
        raise ValueError(f"Wrong window type {config.window_type}")

    tokens = torch.full((batch_size, full_length), config.mask_id, dtype=torch.long, device=device)
    tokens[:, :initial_length] = input_ids
    if hasattr(model, 'generate_init'):
        model.generate_init(tokens)

    cache = None
    average_decode_since_last = float('inf')
    eos_pos = [full_length for _ in range(batch_size)]

    while (tokens == config.mask_id).any():
        pass_cache = False
        if config.cache_type == 'none':
            logits, = model.forward(tokens, None, (window_left, window_right), False)
        elif average_decode_since_last >= config.refresh_count:
            logits, cache = model.forward(tokens, None, (window_left, window_right), True)
            average_decode_since_last = 0
        else:
            pass_cache = True
            if config.cache_type == 'prefix':
                start, stop = window_left, full_length
            elif config.cache_type == 'dual':
                start, stop = window_left, window_right
            # elif config.cache_type == 'prefix-block':
            #     start, stop = initial_length + floor_divisible(window_left - initial_length, block_size), full_length
            # elif config.cache_type == 'dual-block':
            #     start, stop = initial_length + floor_divisible(window_left - initial_length, block_size), initial_length + ceil_divisible(window_right - initial_length, block_size)
            elif config.cache_type.startswith('prefix-delay'):
                w = int(config.cache_type[len('prefix-delay'):])
                start, stop = max(old_window_left - w, initial_length), full_length
            elif config.cache_type.startswith('dual-delay'):
                w = int(config.cache_type[len('dual-delay'):])
                start, stop = max(old_window_left - w, initial_length), min(window_right + w, full_length)
            else:
                raise ValueError(f"Wrong cache type {config.cache_type}")
            logits, cache = model.forward(tokens, cache, (start, stop), True)
            if (start, stop) != (window_left, window_right):
                logits = logits[:, window_left - start:window_right - start]

        confidence, token_generated = sample_tokens(logits, config.temperature, config.kwargs.get('top_p',None),config.kwargs.get('top_k',None),neg_entropy=(config.decode_algo.startswith('entropy')), margin_confidence=(config.decode_algo.startswith('topk_margin')))

        candidate = (tokens[:, window_left:window_right] == config.mask_id)        
        num_decode = get_decode_num(confidence, candidate, config.decode_algo, config.decode_param)
        decode_confidence, decode_indices = get_decode_conf_and_indices(confidence, candidate, num_decode)

        total_decode = 0
        for i in range(batch_size):
            shifted_indices = decode_indices[i] + window_left
            tokens[i, shifted_indices] = token_generated[i, decode_indices[i]]
            decode_indices[i] = shifted_indices
            total_decode += len(decode_indices[i])

            if config.window_type == 'sliding':
                num_mask[i] -= len(decode_indices[i])
            
            if (tokens[i, shifted_indices] == config.eos_id).any():
                eos_first = shifted_indices[tokens[i, shifted_indices] == config.eos_id].min().item()
                if config.window_type == 'sliding':
                    total_decode += (tokens[i, eos_first + 1:eos_pos[i]] == config.mask_id).sum().item()
                    num_mask[i] -= (tokens[i, eos_first + 1:window_right] == config.mask_id).sum().item()
                    tokens[i, eos_first + 1:eos_pos[i]] = config.pad_id
                eos_pos[i] = min(eos_first, eos_pos[i])

                # print(f'[DEBUG] Find EOS at batch {i}/{batch_size} at {eos_first}/{full_length} where window {(window_left, window_right)}. Omitting rest of tokens.')

        if config.debug:
            debug_info.append(OneStepDebugInfo(
                forward_length=model.last_forward_length,
                window_range=(window_left, window_right),
                # token_after_decode=tokens.clone().cpu().int(),
                # token_generated=token_generated.cpu().int(),
                # confidence_generated=confidence.cpu(),
                decode_indices=[x.cpu().int() for x in decode_indices],
                decode_confidence=[x.cpu() for x in decode_confidence],
                pass_cache=pass_cache
            ))

        average_decode_since_last += total_decode / batch_size
        old_window_left, old_window_right = window_left, window_right
        if config.window_type == 'sliding':
            while window_left < tokens.shape[1] and (tokens[:, window_left] != config.mask_id).all():
                window_left += 1
            
            window_right = min(
                window_right + window_size - max(num_mask),
                window_left + max_window_length,
                max(eos_pos)
            )
            delta = window_right - old_window_right
            for i in range(batch_size):
                if eos_pos[i] == full_length:
                    num_mask[i] += delta

        elif (tokens[:, window_left:window_right] != config.mask_id).all():
            # For static window, max_window_length is ignored.
            window_left, window_right = window_right, min(window_right + block_size, full_length)
            average_decode_since_last = float('inf')    # refresh next time
            for i in range(batch_size):
                if eos_pos[i] < full_length:
                    tokens[i, eos_pos[i]:] = config.pad_id
            
            if max(eos_pos) < full_length:
                break
            
    return (tokens, debug_info)

def window_causal_decode(
    model: CausalDLLM,
    input_ids: torch.Tensor,
    config: DecodeConfig
) -> Tuple[torch.Tensor, Optional[List[OneStepDebugInfo]]]:
    ''' This function runs causal diffusion LLM generation in sliding window based decoding strategy.
    '''
    
    generate_length = config.generate_length
    window_size = config.initial_window_length
    block_size = config.block_size
    max_window_length = config.max_window_length
    device = input_ids.device
    batch_size, initial_length = input_ids.shape
    debug_info = [] if config.debug else None

    if model.block_size > 0:
        full_length = floor_divisible(initial_length + generate_length, block_size)
        begin_length = ceil_divisible(initial_length, block_size)
        tokens = torch.full((batch_size, begin_length), config.mask_id, dtype=torch.long, device=device)
        tokens[:, :initial_length] = input_ids
        block_left = (begin_length // block_size - 1) * block_size 
        block_right = block_left + block_size
        cur_block_size = block_size
    else:
        full_length = initial_length + generate_length
        block_left, block_right, cur_block_size = 0, input_ids.shape[1], input_ids.shape[1]
        tokens = input_ids

    def extend_tokens(length):
        nonlocal eos_pos, device, full_length, config, tokens
        new_tokens = torch.where(
            torch.tensor(eos_pos, dtype=torch.long, device=device) == full_length,
            torch.tensor(config.mask_id, dtype=torch.long, device=device),
            torch.tensor(config.pad_id, dtype=torch.long, device=device)
        ).unsqueeze(1).repeat((1, length))
        tokens = torch.cat((tokens, new_tokens), dim=1) 


    if hasattr(model, 'generate_init'):
        model.generate_init(tokens, generate_length)

    prefix_cache, block_cache = None, None
    
    expansion_max = config.kwargs.get('expansion_max', 0)
    expansion_step = config.kwargs.get('expansion_step', 4)
    expansion_threshold = config.kwargs.get('expansion_threshold', 0.3)
    assert expansion_max % expansion_step == 0, "expansion_max must be devided by expansion_step!"

    # debug_block_id = 0
    # debug_decode_id = -1
    # print(f"[DEBUG] Init function with prompt Length = {initial_length}")

    eos_pos = [full_length  for _ in range(batch_size)]
    while max(block_left + 1, initial_length) < full_length:
        '''
        Block i contains 2 stages.

        STAGE #1: Decode all current block
        STAGE #2: Decode first token of next block and update prefix cache
        '''
       
        block_cache = None
        if prefix_cache is None:
            # The first forward pass.
            window_left = initial_length
        else:
            # Enter a new block id.
            window_left = block_left + 1

        # window_left = floor_divisible(window_left, window_size)     # Enable this & window_type = 'static' -> align with FastDLLMv2
        window_right = min(window_left + window_size, block_right)
        # window_left = max(window_left, block_left + 1)
        average_decode_since_last = float('inf')
        if config.window_type == 'sliding':
            num_mask = [0] * batch_size
            for i in range(batch_size):
                if eos_pos[i] == full_length:
                    num_mask[i] = window_right - window_left

        # print(f'[DEBUG] Block {debug_block_id} range {(block_left, block_right)}')
        # debug_block_id += 1

        while True:
            flag = (tokens[:, max(block_left, initial_length):block_right] == config.mask_id).any()
            pass_cache = False

            # def debug_cache(cache):
            #     s = cache.__class__.__name__
            #     if hasattr(cache, 'get_seq_length'):
            #         s += '{' + str(cache.get_seq_length()) + '}'
            #     return s
            # debug_decode_id += 1
            # print(f'[DEBUG] Step {debug_decode_id} before: flag {flag} prefix_cache {debug_cache(prefix_cache)} block_cache {debug_cache(block_cache)} tokens {tokens.shape} window {(window_left, window_right)}.')

            if flag:
                '''
                    STAGE 1:
                    Tokens |TTTT|TTTT|TTTT|TMTM|
                            ^^^^ ^^^^ ^^^^ ~~~~
                            prefix-cache   bl-ca
                '''
                if prefix_cache is None:
                    logits, prefix_cache, _ = model.forward(tokens, (window_left, window_right))
                    prefix_cache, block_cache = split_cache(prefix_cache, block_size)
                    average_decode_since_last = 0
                else:
                    # This part is most important.
                    if average_decode_since_last >= config.refresh_count or config.cache_type == 'none':
                        block_cache = None  # refresh block cache
                        average_decode_since_last = 0
                    pass_cache = (block_cache is not None)
                    if block_cache is None or config.window_type == 'static' or config.cache_type == 'dual':
                        logits, prefix_cache, block_cache = model.forward(
                            tokens, (window_left, window_right),prefix_cache=prefix_cache, block_cache=block_cache 
                        )
                    elif config.cache_type.startswith('dual-delay'):
                        w = int(config.cache_type[len('dual-delay'):])
                        start, stop = max(old_window_left - w, block_left + 1), min(window_right + w, block_right)
                        logits, prefix_cache, block_cache = model.forward(
                            tokens, (start, stop), prefix_cache=prefix_cache, block_cache=block_cache 
                        )
                        if (start, stop) != (window_left, window_right):
                            logits = logits[:, window_left - start:window_right - start]
                    else:
                        raise ValueError(f"Wrong cache type {config.cache_type}")        
            elif max(eos_pos) < full_length or block_right >= full_length:
                break   # Already generate all
            else:
                '''
                    STAGE 2:
                    Tokens |TTTT|TTTT|TTTT|TTTT|
                            ^^^^ ^^^^ ^^^^ ^^^^
                                prefix-cache
                '''
                window_left, window_right = block_right, block_right+1
                logits, prefix_cache, block_cache = model.forward(tokens, (window_left, window_right), prefix_cache=prefix_cache, merge_block=True)
                extend_tokens(min(block_size, full_length - block_right))

            # print(f'[DEBUG] Step {debug_decode_id} after: flag {flag} prefix_cache {debug_cache(prefix_cache)} block_cache {debug_cache(block_cache)} tokens {tokens.shape} window {(window_left, window_right)}.')

            confidence, token_generated = sample_tokens(logits, config.temperature, config.kwargs.get('top_p',None),config.kwargs.get('top_k',None),neg_entropy=(config.decode_algo.startswith('entropy')), margin_confidence=(config.decode_algo.startswith('topk_margin')))

            candidate = (tokens[:, window_left:window_right] == config.mask_id)
            num_decode = get_decode_num(confidence, candidate, config.decode_algo, config.decode_param)
            decode_confidence, decode_indices = get_decode_conf_and_indices(confidence, candidate, num_decode)

            #### extention:
            if config.window_type == 'sliding' and window_right == block_right and flag \
                and cur_block_size - block_size < expansion_max and min(torch.min(x) for x in decode_confidence) < expansion_threshold \
                and min(window_right + window_size - max(num_mask),
                    window_left + max_window_length,
                    max(eos_pos)) > block_right:
                
                step_size = min(expansion_step, full_length - block_right)
                # print(f'[DEBUG] Step {debug_decode_id} end: confidence too low near boundary. Abort it and expand block size by {step_size}.')
                block_right += step_size
                cur_block_size += step_size
                average_decode_since_last = float('+inf')
                window_right = min(
                    window_right + window_size - max(num_mask),
                    window_left + max_window_length,
                    block_right,
                    max(eos_pos)
                )
                extend_tokens(step_size)
                if config.debug:
                    debug_info.append(OneStepDebugInfo(
                        forward_length=model.last_forward_length,
                        window_range=(window_left, window_right),
                        # token_after_decode=tokens.clone().cpu().int(),
                        # token_generated=token_generated.cpu().int(),
                        # confidence_generated=confidence.cpu(),
                        decode_indices=[x.cpu().int() for x in decode_indices],
                        decode_confidence=[x.cpu() for x in decode_confidence],
                        pass_cache=False,
                        reserved_field='abort'
                    ))
                continue


            total_decode = 0
            for i in range(batch_size):
                shifted_indices = decode_indices[i] + window_left
                tokens[i, shifted_indices] = token_generated[i, decode_indices[i]]
                decode_indices[i] = shifted_indices
                total_decode += len(decode_indices[i])

                if config.window_type == 'sliding':
                    num_mask[i] -= len(decode_indices[i])
                
                if (tokens[i, shifted_indices] == config.eos_id).any():
                    eos_first = shifted_indices[tokens[i, shifted_indices] == config.eos_id].min().item()
                    if config.window_type == 'sliding':
                        total_decode += (tokens[i, eos_first + 1:eos_pos[i]] == config.mask_id).sum().item()
                        num_mask[i] -= (tokens[i, eos_first + 1:window_right] == config.mask_id).sum().item()
                        tokens[i, eos_first + 1:eos_pos[i]] = config.pad_id
                    eos_pos[i] = min(eos_first, eos_pos[i])

                    # print(f'[DEBUG] Step {debug_decode_id}: find EOS at batch {i}/{batch_size} at {eos_first}/{full_length} where window {(window_left, window_right)}. Omitting rest of tokens.')

            # print(f'[DEBUG] Step {debug_decode_id} end: decode tokens {decode_indices} with confidence {decode_confidence}, block {tokens[:, block_left:block_right]}.')

            if config.debug:
                debug_info.append(OneStepDebugInfo(
                    forward_length=model.last_forward_length,
                    window_range=(window_left, window_right),
                    # token_after_decode=tokens.clone().cpu().int(),
                    # token_generated=token_generated.cpu().int(),
                    # confidence_generated=confidence.cpu(),
                    decode_indices=[x.cpu().int() for x in decode_indices],
                    decode_confidence=[x.cpu() for x in decode_confidence],
                    pass_cache=pass_cache
                ))

            if not flag:
                break

            average_decode_since_last += total_decode / batch_size
            old_window_left, old_window_right = window_left, window_right
            if config.window_type == 'sliding':
                while window_left < block_right and (tokens[:, window_left] != config.mask_id).all():
                    window_left += 1
                
                window_right = min(
                    window_right + window_size - max(num_mask),
                    window_left + max_window_length,
                    block_right,
                    max(eos_pos)
                )
                delta = window_right - old_window_right
                for i in range(batch_size):
                    if eos_pos[i] == full_length:
                        num_mask[i] += delta
            elif window_right < block_right and (tokens[:, window_left:window_right] != config.mask_id).all():
                # For static window, max_window_length is ignored.
                window_left, window_right = window_right, min(window_right + window_size, block_right)
                average_decode_since_last = float('inf')    # refresh next time

        if max(eos_pos) < full_length:
            break

        block_left, block_right = block_right, min(block_right + block_size, full_length)
        cur_block_size = block_right - block_left
    
    # Remove EOS 
    for i in range(batch_size):
        if eos_pos[i] < full_length:
            tokens[i, eos_pos[i]:] = config.pad_id
    return (tokens, debug_info)

if __name__ == '__main__':
    # Test models here
    pass