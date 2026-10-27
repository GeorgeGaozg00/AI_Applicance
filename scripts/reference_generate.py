#!/usr/bin/env python3
"""CPU reference for diagnosis of NPU inference text; no API/server."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'npu:0'], default='cpu')
    parser.add_argument('--prompt', default='请用中文介绍北京的历史、文化和主要景点，至少写150字。')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    dtype = torch.float32 if args.device == 'cpu' else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                local_files_only=True, trust_remote_code=False, attn_implementation='eager').to(args.device).eval()
    prompt = args.prompt
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt').to(args.device)
    if args.device.startswith('npu'):
        torch.npu.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        result = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    if args.device.startswith('npu'):
        torch.npu.synchronize()
    report = {'backend': f'transformers_{args.device}_{dtype}_reference', 'prompt': prompt,
              'parameter_device': str(next(model.parameters()).device),
              'response': tokenizer.decode(result[0, inputs.input_ids.shape[1]:].cpu(), skip_special_tokens=True),
              'input_tokens': int(inputs.input_ids.shape[1]),
              'elapsed_seconds': time.monotonic() - start, 'max_new_tokens': args.max_new_tokens}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
