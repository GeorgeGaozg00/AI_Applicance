#!/usr/bin/env python3
"""Independent host-driver check; not a production inference server."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DO_NOT_TRACK='1',
                  HF_HUB_DISABLE_TELEMETRY='1', ASCEND_RT_VISIBLE_DEVICES='0')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = {'probe': 'host_torchnpu', 'passed': False, 'device': 'npu:0'}
    result = Path(args.output)
    result.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch_npu
        from transformers import AutoModelForCausalLM, AutoTokenizer
        report['versions'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__}
        report['visible_devices'] = torch.npu.device_count()
        if not torch.npu.is_available():
            raise RuntimeError('No NPU available')
        torch.npu.set_device(0)
        a = torch.ones((4, 4), dtype=torch.float16, device='npu:0')
        c = (a @ a).cpu()
        report['matmul_passed'] = bool(torch.equal(c, torch.full_like(c, 4)))
        if not report['matmul_passed']:
            raise RuntimeError('NPU matmul mismatch')
        begin = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                    local_files_only=True, trust_remote_code=False, attn_implementation='eager').to('npu:0').eval()
        torch.npu.synchronize()
        report['load_seconds'] = time.monotonic() - begin
        report['parameter_device'] = str(next(model.parameters()).device)
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': '请只回答：中国的首都是哪里？'}],
                                              tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors='pt').to('npu:0')
        begin = time.monotonic()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        torch.npu.synchronize()
        report['generation_seconds'] = time.monotonic() - begin
        output = outputs[0, inputs.input_ids.shape[1]:]
        report['generated_tokens'] = int(output.numel())
        report['response'] = tokenizer.decode(output.cpu(), skip_special_tokens=True)
        report['passed'] = '北京' in report['response'] and report['parameter_device'].startswith('npu:')
        del model, inputs, outputs, a, c, output
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        report['allocated_bytes_after_cleanup'] = torch.npu.memory_allocated()
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = str(exc)
        import traceback
        traceback.print_exc()
    finally:
        result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
