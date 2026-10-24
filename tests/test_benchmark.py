"""Process-control tests using synthetic child processes, never GPU measurements."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark', ROOT / 'code' / 'benchmark_130m.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'submission.jsonl'
        self.timing = self.output.with_suffix('.timing.json')

    def command(self, *, actual_t4=True, exit_code=0, write_output=True, suffix=''):
        program = f'''import json, pathlib, sys, time
p = pathlib.Path({str(self.output)!r})
if {write_output!r}: p.write_text('{{"code": "test fixture"}}\\n', encoding='utf-8')
p.with_suffix('.timing.json').write_text(json.dumps({{'actual_t4': {actual_t4!r}, 'rows': 1919, 'parameters': 130183936}}))
print('synthetic child', flush=True)
{suffix}
sys.exit({exit_code})
'''
        return [sys.executable, '-u', '-c', program]

    def run_child(self, command, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return benchmark.run_benchmark(command, self.output, **kwargs)

    def report(self):
        return json.loads(self.timing.read_text(encoding='utf-8'))

    def test_success_includes_full_process_and_log(self):
        result = self.run_child(self.command())
        self.assertTrue(result['t4_limit_verified'])
        self.assertEqual(result['status'], 'completed')
        self.assertGreater(result['cold_subprocess_seconds'], 0)
        self.assertIn('synthetic child', self.output.with_suffix('.log').read_text())

    def test_failed_child_cannot_verify_even_with_valid_report(self):
        with self.assertRaises(RuntimeError):
            self.run_child(self.command(exit_code=7))
        self.assertFalse(self.report()['t4_limit_verified'])
        self.assertEqual(self.report()['process_returncode'], 7)

    def test_other_gpu_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_child(self.command(actual_t4=False))
        self.assertFalse(self.report()['t4_limit_verified'])

    def test_other_gpu_debug_run_never_verifies_t4(self):
        result = self.run_child(self.command(actual_t4=False), allow_other_gpu=True)
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['t4_limit_verified'])

    def test_timeout_retains_report_and_log_without_verifying(self):
        with self.assertRaises(RuntimeError):
            self.run_child(self.command(suffix='time.sleep(10)'), timeout=0.5)
        result = self.report()
        self.assertTrue(result['timed_out'])
        self.assertFalse(result['t4_limit_verified'])
        self.assertEqual(result['status'], 'failed')

    def test_missing_submission_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_child(self.command(write_output=False))
        self.assertFalse(self.report()['t4_limit_verified'])

    def test_stale_outputs_removed_before_failed_rerun(self):
        self.output.write_text('stale')
        self.timing.write_text('{"actual_t4": true}')
        with self.assertRaises(RuntimeError):
            self.run_child([sys.executable, '-c', 'raise RuntimeError("test failure")'])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.report()['t4_limit_verified'])
        self.assertIn('test failure', self.output.with_suffix('.log').read_text())

    def test_invalid_report_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_child(self.command(suffix="p.with_suffix('.timing.json').write_text('broken')"))
        self.assertIn('Invalid inference report', self.report()['error'])

    def test_missing_executable_retains_failure_report(self):
        with self.assertRaises(RuntimeError):
            self.run_child([str(Path(self.directory.name) / 'absent.exe')])
        self.assertFalse(self.report()['t4_limit_verified'])
        self.assertIn('FileNotFoundError', self.report()['error'])

    def test_outer_wall_clock_enforces_180_seconds(self):
        with patch.object(benchmark.time, 'perf_counter', side_effect=[0, 181]):
            with self.assertRaises(RuntimeError):
                self.run_child(self.command())
        self.assertFalse(self.report()['t4_limit_verified'])
        self.assertEqual(self.report()['cold_subprocess_seconds'], 181)


if __name__ == '__main__':
    unittest.main()
