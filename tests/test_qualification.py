import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import io
import threading
import urllib.error

from appliance.cli import (parse_devices, parse_npu_processes, require_idle_npus,
                           validate_model, smoke, container_config, repeated_prose)


class QualificationTests(unittest.TestCase):
    def test_physical_and_logical_device_ids_are_distinct(self):
        text = '''| 1       310P3                 | OK | NA |
| 0       6                     | 0000:82:00.0 | 0 1602 / 44278 |
| 1       310P3                 | OK | NA |
| 1       7                     | 0000:82:00.0 | 0 1318 / 43693 |'''
        devices = parse_devices(text)
        self.assertEqual([d['device_id'] for d in devices], [6, 7])
        self.assertEqual([d['card_id'] for d in devices], [1, 1])
        self.assertEqual(devices[0]['memory_total_mb'], 44278)

    def test_permission_error_is_not_a_healthy_inventory(self):
        self.assertEqual(parse_devices('dcmi module initialize failed. ret is -8005'), [])

    def test_running_workload_is_rejected_without_stopping_it(self):
        output = '''| 4       310P3 | OK | NA |
| 0       0 | 0000:01:00.0 | 0 16340 / 44278 |
| NPU Chip | Process id | Process name | Process memory(MB) |
| 4       0 | 23326 | VLLMEngineCor | 14871 |'''
        self.assertEqual(parse_npu_processes(output)[0]['pid'], 23326)
        with patch('appliance.cli.checked', return_value=output) as command:
            with self.assertRaisesRegex(RuntimeError, 'active workloads'):
                require_idle_npus()
            self.assertEqual(command.call_count, 1)

    def test_failed_engine_does_not_wait_for_full_readiness_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url='http://127.0.0.1:18080', wait=300,
                prompt='首都', expect='北京', watch_container=True,
                output=str(Path(directory) / 'failed.json'))
            with patch('appliance.cli.urllib.request.urlopen', side_effect=urllib.error.URLError('refused')), \
                 patch('appliance.cli.container_config', return_value={'State': {'Status': 'exited', 'ExitCode': 1}}), \
                 patch('appliance.cli.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'Engine unavailable'):
                    smoke(args)
                sleep.assert_not_called()
                self.assertFalse(json.loads(Path(args.output).read_text())['passed'])

    def test_qualification_prose_repetition_is_not_a_quality_pass(self):
        self.assertIsNotNone(repeated_prose('北京是中国的首都，拥有悠久历史。' * 4))
        self.assertIsNone(repeated_prose('北京是中国的首都，拥有悠久历史。故宫和天坛是著名景点。'))

    def test_unowned_container_or_docker_permission_error_is_not_modified(self):
        with patch('appliance.cli.run', return_value={
                'ok': True, 'stdout': json.dumps([{'Config': {'Labels': {'owner': 'other'}}}]),
                'stderr': ''}) as command:
            with self.assertRaisesRegex(RuntimeError, 'Refusing'):
                container_config()
            self.assertEqual(command.call_count, 1)
        with patch('appliance.cli.run', return_value={
                'ok': False, 'stdout': '', 'stderr': 'permission denied'}):
            with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                container_config()

    def test_docker_29_lowercase_missing_container_is_recognized(self):
        with patch('appliance.cli.run', return_value={
                'ok': False, 'stdout': '', 'stderr': 'error: no such object: ai-appliance-v01-model'}):
            self.assertIsNone(container_config())

    def test_tampered_or_traversing_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('config.json', 'tokenizer_config.json', 'model.safetensors'):
                (root / name).write_bytes(b'{}')
            files = [{'path': name, 'size': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}
                     for name in ('config.json', 'tokenizer_config.json', 'model.safetensors')]
            manifest = root / 'appliance-manifest.json'
            manifest.write_text(json.dumps({'files': files}))
            self.assertEqual(validate_model(root), root.resolve())
            (root / 'model.safetensors').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                validate_model(root)
            files[0]['path'] = '../config.json'
            manifest.write_text(json.dumps({'files': files}))
            with self.assertRaises(ValueError):
                validate_model(root)

    def test_stream_and_normal_response_and_truncation(self):
        class Handler(BaseHTTPRequestHandler):
            complete = True

            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': 'qualification'}]}).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.end_headers()
                if body['stream']:
                    item = {'choices': [{'delta': {'content': '北京'}}]}
                    self.wfile.write(('data: ' + json.dumps(item) + '\n\n').encode())
                    if self.complete:
                        self.wfile.write(b'data: [DONE]\n\n')
                else:
                    item = {'choices': [{'message': {'content': '北京'}}]}
                    self.wfile.write(json.dumps(item).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                args = SimpleNamespace(base_url=f'http://127.0.0.1:{server.server_port}',
                                       wait=5, prompt='首都', expect='北京',
                                       output=str(Path(directory) / 'result.json'))
                self.assertEqual(smoke(args), 0)
                report = json.loads(Path(args.output).read_text())
                self.assertTrue(report['non_streaming']['passed'])
                Handler.complete = False
                self.assertEqual(smoke(args), 1)
                self.assertFalse(json.loads(Path(args.output).read_text())['stream_completed'])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == '__main__':
    unittest.main()

