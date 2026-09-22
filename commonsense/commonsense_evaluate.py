import argparse
import copy
import json
import math
import os
import re
import sys
from os.path import join
from pathlib import Path
from typing import List, Optional, Union

import fire
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from tqdm import tqdm

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from datasets import load_dataset

import transformers
from transformers import (
    GenerationConfig,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MixtralAdapterForCausalLM = OlmoeAdapterForCausalLM = Qwen3MoeAdapterForCausalLM = None
def _try_register(conf_mod, conf_cls, model_mod, model_cls, causal_cls, tag):
    try:
        import importlib
        cm = importlib.import_module(conf_mod); mm = importlib.import_module(model_mod)
        conf = getattr(cm, conf_cls)
        AutoConfig.register(tag, conf)
        AutoModel.register(conf, getattr(mm, model_cls))
        AutoModelForCausalLM.register(conf, getattr(mm, causal_cls))
        return getattr(mm, causal_cls)
    except Exception as _e:
        print(f"[skip] {conf_mod}: {type(_e).__name__}: {_e}")
        return None

MixtralAdapterForCausalLM = _try_register(
    "mixtral_modification.configuration_mixtral", "MixtralAdapterConfig",
    "mixtral_modification.modeling_mixtral", "MixtralAdapterModel",
    "MixtralAdapterForCausalLM", "mixtral-adapter")
OlmoeAdapterForCausalLM = _try_register(
    "olmoe_modification.configuration_olmoe", "OlmoeAdapterConfig",
    "olmoe_modification.modeling_olmoe", "OlmoeAdapterModel",
    "OlmoeAdapterForCausalLM", "olmoe-adapter")
Qwen3MoeAdapterForCausalLM = _try_register(
    "qwen3_modification.configuration_qwen3_moe", "Qwen3MoeAdapterConfig",
    "qwen3_modification.modeling_qwen3_moe", "Qwen3MoeAdapterModel",
    "Qwen3MoeAdapterForCausalLM", "qwen3-moe-adapter")


from utils import (
    get_adapter_args,
    init_trainable_parameters,
    convert_trainable_parameters,
    print_trainable_parameters,
)


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
        base_model: str = "mistralai/Mixtral-8x7B-v0.1",
        peft_model: str = "",
):
    args = parse_args()

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=args.max_new_tokens,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            #do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                tokenizer=tokenizer,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        #print(f"outputs: {outputs}")
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return prompts,outputs

    save_file = f'experiment/{args.name}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        prompts,outputs = evaluate(instructions)

        for data, prompt, output in zip(batch, prompts, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['prompt'] = prompt
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            #print(data["instruction"])
            #print(output)
            #print('prediction:', predict)
            #print('label:', label)
        
        pbar.set_description(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}', refresh=True)    
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                the correct answer is"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                the correct answer is"""  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--peft_model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=32)    

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.base_model}')
    peft_model = args.peft_model
    if not peft_model:
        raise ValueError(f'can not find peft weight, the value is: {peft_model}')
    
    
    def set_dropout_to_zero(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    set_dropout_to_zero(value)
                elif isinstance(key, str) and 'dropout' in key.lower():
                    obj[key] = 0.0
        elif isinstance(obj, list):
            for item in obj:
                set_dropout_to_zero(item)
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if isinstance(value, (dict, list)):
                    set_dropout_to_zero(value)
                elif isinstance(key, str) and 'dropout' in key.lower():
                    setattr(obj, key, 0.0)    
    config_path = os.path.join(
        peft_model, "config.json"
    )
    config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    set_dropout_to_zero(config)
    print(config)
    config.output_router_logits = False
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if 'Qwen3' in base_model:
        model = Qwen3MoeAdapterForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
    elif 'Mixtral' in base_model:
        model = MixtralAdapterForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
    elif 'OLMoE' in base_model:
        model = OlmoeAdapterForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    print(tokenizer)

    print(f"##### Loading checkpoint from {peft_model} #####")
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        peft_model, "model.safetensors"
    )  # Full checkpoint
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = load_file(checkpoint_name)

        def load_trainable_params(model, state_dict):
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}# and model_dict[k].requires_grad}
            print(f'##### Loading {len(filtered_dict.keys())} parameters #####')
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
            return model

        model = load_trainable_params(model, adapters_weights)
        print(f"##### Successfully loaded parameters from {checkpoint_name} #####")

        import re as _re
        _pat = ("deq_routing_adapter", "shared_routing_adapter", "shared_adapter",
                "embedded_routing_adapter", "lora_A")
        _trained = set()
        for _k in adapters_weights.keys():
            if any(_p in _k for _p in _pat):
                _mm = _re.search(r"layers\.(\d+)\.", _k)
                if _mm:
                    _trained.add(int(_mm.group(1)))
        if _trained:
            _stripped = 0
            for _i, _layer in enumerate(model.model.layers):
                if _i in _trained:
                    continue
                for _sub in ("mlp", "block_sparse_moe"):
                    _m2 = getattr(_layer, _sub, None)
                    if _m2 is None:
                        continue
                    if getattr(_m2, "deq_routing_adapter", None) is not None:
                        _m2.deq_routing_adapter = None; _stripped += 1
                    if getattr(_m2, "shared_routing_adapter", None):
                        _m2.shared_routing_adapter = None
                        if getattr(_m2, "shared_routing_adapter_gate", None) is not None:
                            _m2.shared_routing_adapter_gate = None
                        _stripped += 1
                    if getattr(_m2, "shared_adapter", None):
                        _m2.shared_adapter = None; _stripped += 1
                    for _e in getattr(_m2, "experts", []):
                        if getattr(_e, "embedded_routing_adapter", None):
                            _e.embedded_routing_adapter = None; _stripped += 1
            if _stripped:
                print(f"##### Sparse checkpoint: adapters trained only in layers {sorted(_trained)}; "
                      f"stripped {_stripped} untrained adapter modules #####")
    else:
        print(f"##### Checkpoint {checkpoint_name} not found #####")
    
    # DEQ_EVAL_INTERVENE:
    _iv = os.environ.get("DEQ_EVAL_INTERVENE", "")
    if _iv:
        n_hit = 0
        for _m in model.modules():
            if not hasattr(_m, "deq_routing_adapter_alpha"):
                continue
            n_hit += 1
            if _iv == "alpha0":
                _m.deq_routing_adapter_alpha.data.zero_()
            elif _iv == "alpha_flip":
                _m.deq_routing_adapter_alpha.data.neg_()
            elif _iv in ("zero_m", "shuffle_m"):
                def _wrap(orig, mode):
                    def fwd(expert_outputs, x, base_weights, m_prev=None):
                        if m_prev is not None:
                            if mode == "zero_m":
                                m_prev = None
                            else:
                                perm = torch.randperm(m_prev.shape[0], device=m_prev.device)
                                m_prev = m_prev[perm]
                        return orig(expert_outputs, x, base_weights, m_prev=m_prev)
                    return fwd
                _m.forward = _wrap(_m.forward, _iv)
            else:
                raise ValueError(f"unknown DEQ_EVAL_INTERVENE={_iv}")
        print(f"##### DEQ_EVAL_INTERVENE={_iv} applied to {n_hit} cross-layer adapters #####")
        assert n_hit > 0, "checkpoint has no cross-layer alpha; intervention is meaningless"

    _ni = os.environ.get("DEQ_EVAL_NITER", "")
    if _ni != "":
        _n_set = 0
        for _m in model.modules():
            if hasattr(_m, "n_iter") and hasattr(_m, "deq_routing_adapter_T_w1"):
                _m.n_iter = int(_ni); _n_set += 1
        print(f"##### DEQ_EVAL_NITER={_ni} applied to {_n_set} adapters #####")
        assert _n_set > 0, "checkpoint has no DEQ adapter; overriding n_iter is meaningless"

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32" and not os.environ.get("NO_TORCH_COMPILE"):
        model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
