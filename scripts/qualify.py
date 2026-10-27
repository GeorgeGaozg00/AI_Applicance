#!/usr/bin/env python3
"""Run on the target host after pulling the pinned image; persist each stage."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from appliance.cli import NAME, run, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--reports', required=True)
    args = p.parse_args()
    folder = Path(args.reports).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    summary = {'version': '0.1.0', 'status': 'running', 'image': args.image,
               'acceptance_schema': 3, 'quality_gate': 'long_prose_repetition_v1',
               'reload_gate': 'same_prompt_and_exact_greedy_response',
               'model_path': str(Path(args.model).resolve()), 'device': 0,
               'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'stages': []}

    def save():
        write_json(folder / 'qualification.json', summary)

    def stage(name, command, timeout=120):
        start = time.monotonic()
        result = run(command, timeout)
        summary['stages'].append({'name': name, 'seconds': time.monotonic() - start, **result})
        save()
        print(name + ': ' + ('PASS' if result['ok'] else 'FAIL'), flush=True)
        if not result['ok']:
            raise RuntimeError(name + ': ' + (result['stderr'] or result['stdout'])[-3000:])
        return result

    cli = [sys.executable, '-m', 'appliance.cli']
    try:
        stage('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', str(repo / 'tests'), '-v'])
        stage('inventory_before', cli + ['doctor', '--output', str(folder / 'inventory-before.json')], 180)
        stage('load_model', cli + ['start', '--model', args.model, '--image', args.image])
        stage('normal_and_streaming', cli + ['smoke', '--wait', '300', '--watch-container', '--output', str(folder / 'smoke-first.json')], 650)
        stage('runtime_versions', ['docker', 'exec', NAME, 'python3', '-c',
              "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ['vllm','vllm-ascend','torch','torch-npu','transformers']}))"])
        for index in range(3):
            stage(f'baseline_{index + 1}', cli + ['smoke', '--wait', '10', '--prompt',
                  '请用中文介绍北京的历史、文化和主要景点，至少写150字。', '--expect', '北京',
                  '--reject-repetition',
                  '--output', str(folder / f'baseline-{index + 1}.json')], 400)
        stage('inventory_loaded', cli + ['doctor', '--output', str(folder / 'inventory-loaded.json')], 180)
        stage('stop_model', cli + ['stop', '--logs', str(folder / 'engine-first.log')])
        stage('inventory_after_stop', cli + ['doctor', '--output', str(folder / 'inventory-after-stop.json')], 180)
        stage('reload_model', cli + ['start', '--model', args.model, '--image', args.image])
        stage('reload_normal_and_streaming', cli + ['smoke', '--wait', '300', '--watch-container',
              '--output', str(folder / 'smoke-reload.json')], 650)
        first = json.loads((folder / 'smoke-first.json').read_text())
        reloaded = json.loads((folder / 'smoke-reload.json').read_text())
        consistent = (first['request'] == reloaded['request'] and
                      first['response'] == reloaded['response'] and
                      first['non_streaming']['text'] == reloaded['non_streaming']['text'])
        summary['reload_consistency'] = {'same_request_and_responses': consistent}
        save()
        if not consistent:
            raise RuntimeError('Reload changed the deterministic response for the same prompt')
        stage('final_stop', cli + ['stop', '--logs', str(folder / 'engine-reload.log')])
        stage('inventory_final', cli + ['doctor', '--output', str(folder / 'inventory-final.json')], 180)
        before = {d['device_id']: d['memory_used_mb'] for d in json.loads(
            (folder / 'inventory-before.json').read_text())['devices']}
        loaded = {d['device_id']: d['memory_used_mb'] for d in json.loads(
            (folder / 'inventory-loaded.json').read_text())['devices']}
        after = {d['device_id']: d['memory_used_mb'] for d in json.loads(
            (folder / 'inventory-final.json').read_text())['devices']}
        allocation = {str(i): loaded[i] - before[i] for i in before}
        released = {str(i): after[i] - before[i] for i in before}
        isolated = allocation['0'] > 100 and all(delta < 256 for i, delta in allocation.items() if i != '0')
        freed = all(delta < 512 for delta in released.values())
        summary['resource_check'] = {'allocation_delta_mb': allocation, 'final_delta_mb': released,
                                     'isolated_device_0': isolated, 'released_within_512mb': freed}
        if not isolated or not freed:
            raise RuntimeError('Resource isolation or memory release check failed; inspect snapshots')
        summary['status'] = 'passed'
    except (OSError, RuntimeError) as exc:
        summary['status'] = 'failed'
        summary['error'] = str(exc)
        # Preserve diagnostics; do not silently call a failed model a deployed service.
        diagnostic = run(['docker', 'logs', '--tail', '200', NAME])
        (folder / 'failure-engine.log').write_text(diagnostic['stdout'] + diagnostic['stderr'])
        print(summary['error'], file=sys.stderr, flush=True)
    finally:
        summary['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())

