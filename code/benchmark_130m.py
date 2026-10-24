"""Time a new inference process; retain diagnostics even on failure or timeout."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def run_benchmark(command, output, *, timeout=190, allow_other_gpu=False):
    output = Path(output)
    timing = output.with_suffix('.timing.json')
    log = output.with_suffix('.log')
    # A failed rerun must never expose a previous submission as its output.
    timing.unlink(missing_ok=True)
    output.unlink(missing_ok=True)
    started = time.perf_counter()
    returncode = None
    error = None
    timed_out = False
    with log.open('w', encoding='utf-8') as stream:
        try:
            process = subprocess.run(command, timeout=timeout, stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error = f'Inference exceeded the {timeout:g}-second process timeout'
        except OSError as exc:
            error = f'{type(exc).__name__}: {exc}'
    seconds = time.perf_counter() - started
    result = {}
    if timing.exists():
        try:
            result = json.loads(timing.read_text(encoding='utf-8'))
            if not isinstance(result, dict):
                raise ValueError('Expected a JSON object')
        except (ValueError, OSError) as exc:
            result = {}
            error = f'Invalid inference report: {exc}'
    complete = (returncode == 0 and not timed_out and error is None
                and output.is_file() and result.get('rows') == 1919
                and result.get('parameters') == 130_183_936)
    verified = complete and result.get('actual_t4') is True and seconds <= 180
    success = complete and (verified or allow_other_gpu)
    if not success and error is None:
        error = ('The real T4 run exceeded 180 seconds or used a different GPU'
                 if complete else f'Inference failed or did not write complete output (exit {returncode})')
    result.update(status='completed' if success else 'failed',
                  cold_subprocess_seconds=seconds, process_returncode=returncode,
                  timed_out=timed_out, process_timeout_seconds=timeout,
                  t4_limit_verified=verified, within_180_seconds=complete and seconds <= 180,
                  log_file=str(log), error=error)
    timing.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if not success:
        raise RuntimeError(f'{error}. See {log} and {timing}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoint_130m.zip')
    parser.add_argument('--questions', default='questions_used.jsonl')
    parser.add_argument('--output', default='submission_130m.jsonl')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--timeout', type=float, default=190)
    parser.add_argument('--allow-other-gpu', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    command = [sys.executable, '-u', str(Path(__file__).with_name('inference_130m.py')),
               '--checkpoint', args.checkpoint, '--questions', args.questions,
               '--output', args.output, '--batch-size', str(args.batch_size),
               '--max-new-tokens', str(args.max_new_tokens)]
    if args.allow_other_gpu:
        command.append('--allow-other-gpu')
    run_benchmark(command, args.output, timeout=args.timeout, allow_other_gpu=args.allow_other_gpu)


if __name__ == '__main__':
    main()
