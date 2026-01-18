import logging
import os
from datetime import timedelta
from typing import Dict, List, Literal, Optional, Tuple, Union, TypeVar
import torch
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from functools import partial
from datasets import Dataset
from accelerate.utils import get_max_memory
from packaging import version
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype, configure_pad_token
from lm_eval.__main__ import cli_evaluate

from dllm_model import Llada, Dream, FastDLLMv2, BidirectionalDLLM, CausalDLLM, NAME2PATH, NAME2CLASS
from decode_algorithm import DecodeConfig, OneStepDebugInfo, window_bidirectional_decode, window_causal_decode
from visualization import analyze_debug_infos

eval_logger = logging.getLogger(__name__)

# This class is generally based on https://github.com/Li-Jinsong/DAEDAL
@register_model("model")
class Model(LM):

    # For BidirectionalDLLM and CausalDLLM, just test the correctness of workflow.
    # These two models simulate the normal DLLM models along with qwen-0.5B tokenizer.
    SUPPORTED_MODELS = list(NAME2PATH.keys())

    def __init__(
        self,
        model_type: str = 'Llada',  # Llada, Dream, FastDLLMv2
        lookahead: int = 0,
        batch_size: int = 8,
        window_type: str = 'sliding',
        generate_length: int = 512,
        initial_window_length: int = 32,
        block_size: int = 32,
        temperature: float = 0.0,
        decode_algo: str = 'threshold',
        decode_param: Union[int, float] = 0.9,
        max_window_length: int = 1000000000,
        cache_type: str = 'none',
        refresh_count: int = 16,
        debug_dir: str = '.',
        debug_type: str = 'none',
        **kwargs
    ) -> None:
        '''Initialize the model'''
        super().__init__()

        assert model_type in self.SUPPORTED_MODELS, f"Wrong model type {model_type}"
        assert max_window_length >= initial_window_length, "max_window_length must be larger than initial_window_length!"
        self.model_type = model_type
        model_path = NAME2PATH[model_type]

        # none -> don't save any
        # data -> save debug data
        # stat -> save statistics
        # full -> save bot hdebug_data and profile
        self.debug_dir = debug_dir
        self.debug_type = debug_type

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.assistant_prefix = kwargs.pop('assistant_prefix', None)
        self._model = NAME2CLASS[model_type](model_path, lookahead=lookahead, block_size=block_size, **kwargs)
        if model_type == 'Llada':
            special_tokens = {'mask_id': 126336, 'pad_id': 126081, 'eos_id': 126081}
        elif model_type == 'Dream' or model_type == 'DreamBase':
            special_tokens = {'mask_id': 151666, 'pad_id': 151643, 'eos_id': 151643}
        elif model_type == 'FastDLLMv2':
            special_tokens = {'mask_id': 151665, 'pad_id': 151643, 'eos_id': 151645}
        else:
            special_tokens = {'mask_id': 0, 'pad_id': 0, 'eos_id': 0}
        self.config = DecodeConfig(
            window_type=window_type,
            generate_length=generate_length,
            initial_window_length=initial_window_length,
            block_size=block_size,
            temperature=temperature,
            decode_algo=decode_algo,
            decode_param=decode_param,
            max_window_length=max_window_length,
            cache_type=cache_type,
            refresh_count=refresh_count,
            debug=debug_type != 'none',
            **special_tokens
        )
        if isinstance(self._model, BidirectionalDLLM):
            self.gen_func = partial(window_bidirectional_decode, model=self._model, config=self.config)
        elif isinstance(self._model, CausalDLLM):
            self.gen_func = partial(window_causal_decode, model=self._model, config=self.config)
        else:
            raise ValueError(f"Error: model has unsupported type {type(self._model)}")
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        if self.accelerator is not None:
            self.device = torch.device(f'{self.accelerator.device}')
            self._model = self.accelerator.prepare(self._model)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = self._model.to(self.device)
            self._rank = 0
            self._world_size = 1
        
        self.is_first_inference = True

    # The following 2 properties are defined in its parent class:

    # @property
    # def rank(self):
    #     return self._rank

    # @property
    # def world_size(self):
    #     return self._world_size

    @property
    def model(self):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        special_tokens_kwargs = {}
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs["add_special_tokens"] = self.add_bos_token
        else:
            special_tokens_kwargs["add_special_tokens"] = add_special_tokens
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        # add_special_tokens = {"add_special_tokens": self.add_bos_token} if self.backend == "causal" else {}
        add_special_tokens = {}
        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len and encoding["input_ids"].size(1) > left_truncate_len:
            eval_logger.warning(
                f"Left-truncating from {encoding['input_ids'].size(1)} to {left_truncate_len} tokens."
            )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self.tokenizer.padding_side = old_padding_side
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=False):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError
         
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running generate_until requests")
        ds_data = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds_data)

        all_debug_info = []
        for batch in ds.iter(batch_size=int(self.batch_size)):
            contexts = batch["text"]
            context_enc, attn_masks = self.tok_batch_encode(contexts)

            prompt_length = context_enc.shape[1]
            if (context_enc == self.config.mask_id).any():
                if self.model_type in ['BidirectionalDLLM', 'CausalDLLM']:
                    eval_logger.info(f"Mask id accidentally appears in prompts.")
                else:
                    eval_logger.error(f"ERROR! Mask id appears in prompts.")
                    eval_logger.error(f"Number of mask: {(context_enc == self.config.mask_id).sum().item()}")
                    eval_logger.error(f"Context: {contexts} Context_enc: {context_enc}")

                # replace mask_id with another token
                another_token = self.tokenizer.encode(' ')[0]
                if another_token == self.config.mask_id:
                    another_token = self.tokenizer.encode('?')[0]
                context_enc[context_enc == self.config.mask_id] = another_token

            # This is the generation process
            output, debug_info = self.gen_func(input_ids=context_enc)

            cont_toks_list = self.tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)

            if self.rank == 0 and self.is_first_inference:
                eval_logger.info("\n\n--- First Batch Inference (Rank 0) ---")
                for i, (question, answer) in enumerate(zip(contexts, cont_toks_list)):
                    eval_logger.info(f"Question {i+1}: {question}")
                    eval_logger.info(f"\nAnswer   {i+1}: {answer}\n")
                eval_logger.info("------------------------------------\n\n")
                self.is_first_inference = False

            res.extend(cont_toks_list)
            bar.update(len(cont_toks_list))
            all_debug_info.append(debug_info)

        bar.close()

        if self.debug_type == 'full' or self.debug_type == 'data':
            os.makedirs(self.debug_dir, exist_ok=True)
            torch.save(all_debug_info, os.path.join(self.debug_dir, f'debug_{self.rank}.pth'))

        if self.debug_type == 'full' or self.debug_type == 'stat':
            os.makedirs(self.debug_dir, exist_ok=True)
            analyze_debug_infos(all_debug_info, os.path.join(self.debug_dir, f'stat_{self.rank}.json'))

        return res

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.assistant_prefix:
            chat_templated += self.assistant_prefix
        return chat_templated

if __name__ == "__main__":
    cli_evaluate()