from __future__ import annotations

import logging
from typing import List, Tuple, Optional, Iterable, Any
from dataclasses import fields
import torch
from transformers.cache_utils import Cache, DynamicCache
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

from llada.model.modeling_llada import LLaDAModelLM
from dream.model.modeling_dream import DreamModel


class BidirectionalDLLM(nn.Module):
    ''' This is an abstract model for bidirectional diffusion LLM. (e.g, LLADA or DREAM). It provides interfaces for our algorithm.'''
    def __init__(self, shift=0, lookahead=0, **kwargs):
        '''- shift: one position's logits predict next (?) tokens value.
        - lookahead: if shift>0 and predicted tokens are in range [L, R), then logits in positions [L-shift, R-shift) predits them. But to increase attention horizon, positions in [L-shift,R-shift+lookahead) are passed into bottom model. 0<=lookahead<=shift'''
        super().__init__()
        self.last_forward_length = 0
        self.shift = shift
        self.lookahead = lookahead
    
    def forward(self, all_input_ids: torch.Tensor, cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]], predict_range: Tuple[int, int], use_cache: bool = True) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        '''Forward a DLLM model.
        - all_input_ids: torch.Tensor (BATCHSIZE, LENGTH) int64
            All the input_ids in one batch.
        - cache: list(tuple(k_cache, v_cache)) or None, k/v_cache torch.Tensor (BATCHSIZE, HEAD, LENGTH, HIDDEN) bfloat16
            KV cache for all tokens.
        - predict_range [L, R)
            Predict which positions's [MASK] logits.
        - use_cache: if `False`, only return (logits,) torch.Tensor (BATCHSIZE, sum(predict_range)) bfloat16; otherwise, return (logits, new_cache).
        NOTEICE: the new_cache contains all tokens, similar to parameters.
        '''
        # This is a dummy implementation of such model.
        head_num, kv_hidden, vocab_size, layer_num, mask_id = 2, 3, 4, 2, 0
        length_of_predict_range = predict_range[1] - predict_range[0]
        batch_size, length = all_input_ids.shape
        logits = torch.rand(batch_size, length_of_predict_range, vocab_size, device=all_input_ids.device, dtype=torch.bfloat16)
        logits[:,:, mask_id] = -torch.inf

        # last_forward_length is either full length or window length plus lookahead.
        if cache is None:
            self.last_forward_length = length
        else:
            self.last_forward_length = length_of_predict_range + self.lookahead
        
        if not use_cache:
            return (logits, )
        else:
            # The cache if always full cache
            cache = []
            for i in range(layer_num):
                k_cache = torch.randn(batch_size, head_num, length, kv_hidden, device=all_input_ids.device, dtype=torch.bfloat16)
                v_cache = torch.randn(batch_size, head_num, length, kv_hidden, device=all_input_ids.device, dtype=torch.bfloat16)
                cache.append((k_cache, v_cache))
            return (logits, cache)

def split_cache(cache: DynamicCache, length: int) -> Tuple[DynamicCache, DynamicCache]:
    '''Split cache of length L to 2 caches of length (L-l, l)'''
    total_length = cache.get_seq_length()
    assert 0 < length < total_length, "The length if not correct!"
    cache1, cache2 = DynamicCache(), DynamicCache()
    for attr in ['key_cache', 'value_cache']:
        for data in getattr(cache, attr):
            getattr(cache1, attr).append(data[:, :, :-length, :])
            getattr(cache2, attr).append(data[:, :, -length:, :])
    return cache1, cache2

def merge_cache(cache1: DynamicCache, cache2: DynamicCache) -> DynamicCache:
    '''Merge 2 caches'''
    cache = DynamicCache()
    for attr in ['key_cache', 'value_cache']:
        for data1, data2 in zip(getattr(cache1, attr), getattr(cache2, attr)):
            getattr(cache, attr).append(torch.cat((data1, data2), dim=2))
    return cache

class CausalDLLM(nn.Module):
    ''' This is an abstract model for casual diffusion LLM. (e.g, FastDLLM-V2). It provides interfaces for our algorithm.'''
    def __init__(self, shift=1, block_size=2, mask_token_id=0, **kwargs):
        '''- shift: one position's logits predict next (?) tokens value.'''
        super().__init__()
        self.last_forward_length = 0
        self.shift = shift
        self.block_size = block_size
        self.mask_token_id = mask_token_id

    def forward(self, 
                all_input_ids: torch.Tensor,
                predict_range: Tuple[int, int],
                prefix_cache: Optional[DynamicCache] = None,
                merge_block: bool = False,
                block_cache: Optional[DynamicCache] = None) -> Tuple[torch.Tensor, DynamicCache, Optional[DynamicCache]]:
        ''' Forward a DLLM model.
        - all_input_ids: torch.Tensor (BATCHSIZE, LENGTH) int64
            All the input_ids.
        - predict_range: tuple(int, int) [L, R)
            Predict which positions's [MASK] logits.
        - prefix_cache:
            * If None, ignores use_block_cache and block_cache; all_input_ids are sent to model and prefix_cache is filled and returned.
            * If not None, continues.
        - merge_block: whether to merge the current block to prefix_cache
        - block_cache: The block cache (after prefix cache) The replace_position
            The replace_position is calulated automatically based on predict_range

        Return (logits, prefix_cache, block_cache)
        '''
        # This is a dummy implementation of such model.
        head_num, kv_hidden, vocab_size, layer_num, mask_id = 2, 3, 4, 2, 0
        length_of_predict_range = predict_range[1] - predict_range[0]
        batch_size, length = all_input_ids.shape
        logits = torch.rand(batch_size, length_of_predict_range, vocab_size, device=all_input_ids.device, dtype=torch.bfloat16)
        logits[:,:, mask_id] = -torch.inf

        # last_forward_length is either full length or window length plus lookahead.
        if prefix_cache is None:
            self.last_forward_length = length
        else:
            self.last_forward_length = self.block_size
        
        def get_cache(length):
            cache = DynamicCache()
            cache._seen_tokens = length
            for attr in ['key_cache', 'value_cache']:
                setattr(cache, attr, [
                    torch.randn(batch_size, head_num, length, kv_hidden, device=all_input_ids.device, dtype=torch.bfloat16) for _ in range(layer_num)
                ])
            return cache

        if not prefix_cache:
            prefix_cache = get_cache(length)
            return (logits, prefix_cache, None)
        else:
            if block_cache is None:
                block_cache = get_cache(length - prefix_cache.get_seq_length())
            if merge_block:
                prefix_cache = merge_cache(prefix_cache, block_cache)
                block_cache = None
            if self.block_size < 0:
                block_cache = None
            return (logits, prefix_cache, block_cache)

class Llada(BidirectionalDLLM):
    def __init__(self, model_path='GSAI-ML/LLaDA-8B-Instruct', **kwargs):
        super().__init__(0, 0)
        config = AutoConfig.from_pretrained(model_path)
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config)
        self.model.eval()

    def forward(self, all_input_ids, cache, predict_range, use_cache = True):
        window_left, window_right = predict_range
        assert 0 <= window_left < window_right <= all_input_ids.shape[1], f"Bad predict_range {predict_range}"
        # No attention mask are passed into model according to Fast-dLLM implementation.
        # If you really want it, pass (BATCH_SIZE, SEQLEN), all_input_ids.ne(self.model.config.pad_token_id)
        if cache is None:
            self.last_forward_length = all_input_ids.shape[1]
            result = self.model(all_input_ids, use_cache=use_cache)
            logits = result.logits[:, window_left:window_right]
        else:
            self.last_forward_length = window_right - window_left
            replace_position = torch.zeros_like(all_input_ids, dtype=torch.bool)
            replace_position[:, window_left:window_right] = True
            result = self.model(all_input_ids[:, window_left:window_right], use_cache=use_cache, replace_position=replace_position, past_key_values=cache)
            logits = result.logits
        
        logits[:, :, self.model.config.mask_token_id] = -torch.inf  # To avoid generate a `mask`
        return (logits, result.past_key_values) if use_cache else (logits, )

class Dream(BidirectionalDLLM):
    def __init__(self, model_path='Dream-org/Dream-v0-Base-7B', lookahead=1, **kwargs):
        super().__init__(1, lookahead)
        # dream model doesn't need a config to initialize
        self.model = DreamModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.eval()
        self.attention_mask, self.tok_idx = None, None

    def generate_init(self, all_input_ids):
        if all_input_ids.shape[0] > 1:
            attention_mask = all_input_ids.ne(self.model.generation_config.pad_token_id)
            attention_mask = torch.cumsum(all_input_ids, dim=1).bool()
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            attention_mask = 'full'
            tok_idx = None
        self.attention_mask = attention_mask
        self.tok_idx = tok_idx

    def forward(self, all_input_ids, cache, predict_range, use_cache = True):
        window_left, window_right = predict_range
        assert 0 < window_left < window_right <= all_input_ids.shape[1], f"Bad predict_range {predict_range}"
        # According to fast-dLLM implementation, do so.
        # dual_cache=dual_cache, replace_position=replace_position replace_position=(1, SEQLEN) bool
        
        if cache is None:
            self.last_forward_length = all_input_ids.shape[1]
            result = self.model(all_input_ids, attention_mask=self.attention_mask, position_ids=self.tok_idx, use_cache=use_cache)
            logits = result.logits[:, window_left-1:window_right-1]
        else:
            forward_slice = slice(max(window_left-1, 0), window_right-1+self.lookahead)

            attention_mask = self.attention_mask[:, :, forward_slice] if isinstance(self.attention_mask, torch.Tensor) else self.attention_mask
            tok_idx = self.tok_idx[:, forward_slice]  if isinstance(self.tok_idx, torch.Tensor) else self.tok_idx

            self.last_forward_length = forward_slice.stop - forward_slice.start
            replace_position = torch.zeros((1, all_input_ids.shape[1]), dtype=torch.bool, device=all_input_ids.device)
            replace_position[:, forward_slice] = True
            result = self.model(all_input_ids[:, forward_slice], attention_mask=attention_mask, position_ids=tok_idx, use_cache=use_cache, replace_position=replace_position, past_key_values=cache, dual_cache=True)
            logits = result.logits
            if self.lookahead > 0:
                logits = logits[:, :-self.lookahead]

        logits[:, :, self.model.config.mask_token_id] = -torch.inf  # To avoid generate a `mask`
        return (logits, result.past_key_values) if use_cache else (logits, )

class FastDLLMv2(CausalDLLM):
    def __init__(self, model_path='Efficient-Large-Model/Fast_dLLM_v2_7B',block_size=32, **kwargs):
        super().__init__(1, block_size, 151665)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model.eval()
    
    def forward(self, 
                all_input_ids: torch.Tensor,
                predict_range: Tuple[int, int],
                prefix_cache: Optional[DynamicCache] = None,
                merge_block: bool = False,
                block_cache: Optional[DynamicCache] = None) -> Tuple[torch.Tensor, DynamicCache, Optional[DynamicCache]]:
        window_left, window_right = predict_range
        assert 0 < window_left < window_right <= all_input_ids.shape[1]+1, f"Bad predict_range {predict_range}"

        if prefix_cache is None:
            self.last_forward_length = all_input_ids.shape[1]
            result = self.model(all_input_ids, use_cache=True, update_past_key_values=True, block_size=self.block_size)
            logits = result.logits[:, window_left-1:window_right-1]
        else:
            cache_len = prefix_cache.get_seq_length()
            assert window_left > cache_len, f"Cache length = {cache_len} must be less than window left {window_left}!"
            if block_cache is not None:
                forward_slice = slice(window_left-1, window_right-1)
                replace_position = window_left-1-cache_len
            else:
                forward_slice = slice(cache_len, all_input_ids.shape[1])
                replace_position = None
            self.last_forward_length = forward_slice.stop - forward_slice.start
            result = self.model(all_input_ids[:, forward_slice], use_cache=True, update_past_key_values=False, past_key_values=prefix_cache, block_size=self.block_size, replace_position=replace_position, use_block_cache=True, block_past_key_values=block_cache)
            logits = result.logits
            if replace_position is None:
                logits = logits[:, window_left-1-cache_len:window_right-1-cache_len]

        logits[:, :, self.mask_token_id] = -torch.inf  # To avoid generate a `mask`

        # process merge_block
        prefix_cache, block_cache = result.past_key_values, result.block_past_key_values
        if merge_block and (block_cache is not None):
            prefix_cache = merge_cache(prefix_cache, block_cache)
            block_cache = None
        return (logits, prefix_cache, block_cache)

class BlockDynamicCache(DynamicCache):
    """
    When `skip_cache_update` is True, this class does NOT update the cached key and value states.
    Instead, it concatenates the current states with the original cached states along the sequence dimension
    and returns the result. 

    Example:

        ```python
        >>> past_key_values = BlockDynamicCache()
        >>> past_key_values.skip_cache_update = True
        >>> outputs.past_key_values
        ```
    """
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None) -> None:
        """
        Initialize a BlockDynamicCache instance.

        skip_cache_update is False by default.
        """
        super().__init__(_distributed_cache_data)
        self.skip_cache_update = False
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Behavior depends on the `skip_cache_update` flag:
        - If `skip_cache_update` is True:
            * Does NOT update the stored cache.
            * Concatenates the current `key_states` and `value_states` 
              with the original cached states along the sequence dimension.
            * Returns the concatenated result.
        - If `skip_cache_update` is False:
            * Uses the parent class update logic to update the cache.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The updated key and value states after concatenation or update.
                When `skip_cache_update=True`, returns the concatenated tensor without modifying cache.
                When `skip_cache_update=False`, returns the result from the parent class.
        """
        if self.skip_cache_update:
            key_cache = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            value_cache = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            return key_cache, value_cache
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


class NBdiff(CausalDLLM):
    """
    NBdiff: A diffusion-based LLM interface for models like Huawei's openPangu-R-7B-Diffusion.
    Compatible with the CausalDLLM protocol but does not use its internal logic.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",   # or 'npu' on huawei devices
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs  # for compatibility
    ):
        super().__init__(1, 0, 45830)        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            local_files_only=True
        )
        self.model.eval()
        self.device = device

        self.last_forward_length: int = 0
        # Buffers for generate_init
        self._cached_padding_mask: Optional[torch.Tensor] = None
        self._cached_attention_mask: Optional[torch.Tensor] = None
        self._cached_position_ids: Optional[torch.Tensor] = None

    def generate_init(self, all_input_ids: torch.Tensor, max_new_tokens: int = 1024):
        """Precompute attention mask and position IDs for the full sequence."""
        B, L = all_input_ids.shape
        device = all_input_ids.device
        max_length = L + max_new_tokens

        causal_mask = torch.tril(torch.ones(max_length, max_length, device=device, dtype=torch.bool))[None, None, :, :]
        pad_token_id = getattr(self.model.config, 'pad_token_id', 0)
        attention_mask = (all_input_ids != pad_token_id)

        padding_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        position_ids = padding_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(padding_mask == 0, 1)
        # [B, N] --> [B, 1, N, N]
        padding_mask = torch.logical_and(
            padding_mask.unsqueeze(1).unsqueeze(-2),
            padding_mask.unsqueeze(1).unsqueeze(-1),
        )
        attention_mask = padding_mask & causal_mask

        self._cached_position_ids = position_ids    # e.g [0, 1, 2, 3, 4]...
        self._cached_attention_mask = attention_mask    # for batchsize=1, it is standard causal mask
        self._cached_padding_mask = padding_mask    # for batchsize=1, it is all one

    def forward(self, 
                all_input_ids: torch.Tensor,
                predict_range: Tuple[int, int],
                prefix_cache: Optional[DynamicCache] = None,
                merge_block: bool = False,
                block_cache: Optional[DynamicCache] = None) -> Tuple[torch.Tensor, DynamicCache, Optional[DynamicCache]]:
        
        window_left, window_right = predict_range
        assert 0 < window_left < window_right <= all_input_ids.shape[1] + 1, f"Bad predict_range {predict_range}"
        assert block_cache is None, "NBdiff don't support block_cache yet."

        if self._cached_attention_mask is None:
            raise RuntimeError("generate_init() must be called before forward.")

        if prefix_cache is None:
            prefix_cache = BlockDynamicCache()
        prefix_length = prefix_cache.get_seq_length()
        prefix_cache.skip_cache_update = (not merge_block)
        L, R = prefix_length, all_input_ids.shape[1]
        input_slice = all_input_ids[:, L:R]
        if not merge_block: # the block is bidirectional
            attn_mask = self._cached_padding_mask[:, :, L:R, :R]
        else:   # it is causal
            attn_mask = self._cached_attention_mask[:, :, L:R, :R]
        pos_ids = self._cached_position_ids[:, L:R]

        outputs = self.model(
            input_ids=input_slice,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=True,
            past_key_values=prefix_cache
        )
        logits = outputs.logits[:, window_left - 1 - L:window_right - 1 - L]
        self.last_forward_length = R - L
        return_logits = logits
        returned_cache = outputs.past_key_values

        return_logits[:, :, self.mask_token_id] = -torch.inf
        return (return_logits, returned_cache, None)

NAME2PATH = {
    'Llada': 'GSAI-ML/LLaDA-8B-Instruct',
    'Dream': 'Dream-org/Dream-v0-Instruct-7B',
    'DreamBase': 'Dream-org/Dream-v0-Base-7B',
    'FastDLLMv2': 'Efficient-Large-Model/Fast_dLLM_v2_7B',
    'NBdiff': 'path_to_nbdiff',
    'BidirectionalDLLM': 'Qwen/Qwen2.5-0.5B-Instruct',
    'CausalDLLM': 'Qwen/Qwen2.5-0.5B-Instruct',
}

NAME2CLASS = {
    'Llada': Llada,
    'Dream': Dream,
    'DreamBase': Dream,
    'FastDLLMv2': FastDLLMv2,
    'NBdiff': NBdiff,
    'BidirectionalDLLM': BidirectionalDLLM,
    'CausalDLLM': BidirectionalDLLM,
}

NAME2SPECIAL = {
    'Llada': {'mask_id': 126336, 'pad_id': 126081, 'eos_id': 126081},
    'Dream': {'mask_id': 151666, 'pad_id': 151643, 'eos_id': 151643},
    'DreamBase': {'mask_id': 151666, 'pad_id': 151643, 'eos_id': 151643},
    'FastDLLMv2': {'mask_id': 151665, 'pad_id': 151643, 'eos_id': 151645},
    'NBdiff': {'mask_id': 45830, 'pad_id': 0, 'eos_id': 45892},
    'BidirectionalDLLM': {'mask_id': 0, 'pad_id': 0, 'eos_id': 0},
    'ContextCausalDLLM': {'mask_id': 0, 'pad_id': 0, 'eos_id': 0}
}

if __name__ == '__main__':
    pass