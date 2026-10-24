"""Standalone cold-load inference benchmark, strict NVIDIA T4 by default.

Dependency installation and obtaining input artifacts precede the timed run.
The measurement includes imports, checkpoint extraction/loading, tokenization,
GPU generation, and writing/checking the 1919-line output. No datasets library.
"""
import argparse
import json
from pathlib import Path
import time
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoint_130m.zip')
    parser.add_argument('--questions', default='questions_used.jsonl')
    parser.add_argument('--output', default='submission_130m.jsonl')
    parser.add_argument('--allow-other-gpu', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-new-tokens', type=int)
    args = parser.parse_args()
    begin = time.perf_counter()
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('USE_TF', '0')
    os.environ.setdefault('USE_FLAX', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import math
    import hashlib
    import ast
    import platform
    import torch
    from tokenizers import Tokenizer
    from transformers import BartForConditionalGeneration
    import transformers
    phases = {'imports': time.perf_counter() - begin}
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime')
    gpu = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    actual_t4 = gpu in {'Tesla T4', 'NVIDIA Tesla T4', 'NVIDIA T4', 'T4'} and capability == (7, 5)
    if not args.allow_other_gpu and not actual_t4:
        raise RuntimeError(f'This submission benchmark requires an actual T4; found {gpu}')
    torch.cuda.reset_peak_memory_stats()
    phase_started = time.perf_counter()
    archive_path = Path(args.checkpoint)
    checkpoint_sha256 = None
    if archive_path.is_file():
        digest = hashlib.sha256()
        with archive_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        target = Path('checkpoint_130m_extracted')
        with zipfile.ZipFile(archive_path) as archive:
            allowed = {'best/' + n for n in ['config.json', 'generation_config.json', 'model.safetensors',
                       'tokenizer.json', 'training_meta.json', 'inference_config.json']}
            files = [item.filename for item in archive.infolist() if not item.is_dir()]
            directories = {item.filename for item in archive.infolist() if item.is_dir()}
            if set(files) != allowed or len(files) != len(allowed) or not directories <= {'best/'}:
                raise ValueError('Unexpected checkpoint archive files')
            for name in sorted(allowed):
                archive.extract(name, target)
        checkpoint = target / 'best'
    else:
        checkpoint = archive_path
    phases['checkpoint_extraction_and_hash'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    settings = json.loads((checkpoint / 'inference_config.json').read_text(encoding='utf-8'))
    if args.batch_size is not None:
        settings['batch_size'] = args.batch_size
    if args.max_new_tokens is not None:
        settings['max_new_tokens'] = args.max_new_tokens
    if settings['batch_size'] < 1 or not 1 <= settings['max_new_tokens'] <= settings['max_target']:
        raise ValueError('Invalid batch size or generation length')
    questions_bytes = Path(args.questions).read_bytes()
    normalized_questions_sha256 = hashlib.sha256(questions_bytes.replace(b'\r\n', b'\n')).hexdigest()
    if normalized_questions_sha256 != 'a3a128e5acf694f9b9a828dcc582bacb79367eecf11af2f6041c0c5f788e7003':
        raise ValueError('Questions or their order differ from the repository test input')
    rows = [json.loads(line) for line in questions_bytes.decode('utf-8').splitlines()]
    questions = [row['question'] for row in rows]
    if len(questions) != 1919 or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError('Expected 1919 nonempty questions in original order')
    tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
    model = BartForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True, torch_dtype=torch.float16,
        attn_implementation='sdpa').cuda().eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 130_183_936:
        raise ValueError('Wrong model checkpoint')
    if (model.config.pad_token_id, model.config.eos_token_id) != (0, 3):
        raise ValueError('Checkpoint special tokens do not match this inference program')
    torch.cuda.synchronize()
    phases['input_and_model_load'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    encoded = tokenizer.encode_batch(questions, add_special_tokens=False)
    source_clipped = sum(len(x.ids) + 1 > settings['max_source'] for x in encoded)
    sequences = [x.ids[:settings['max_source'] - 1] + [3] for x in encoded]
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    predictions = [None] * len(questions)
    no_eos = 0
    batch_size = settings['batch_size']
    phases['tokenization'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            width = math.ceil(max(len(sequences[i]) for i in indices) / 8) * 8
            ids = torch.zeros((len(indices), width), dtype=torch.long)
            for j, index in enumerate(indices):
                ids[j, :len(sequences[index])] = torch.tensor(sequences[index], dtype=torch.long)
            ids = ids.to('cuda')
            generated = model.generate(input_ids=ids, attention_mask=ids.ne(0).long(),
                                       do_sample=False, num_beams=1, use_cache=True,
                                       max_new_tokens=settings['max_new_tokens'], forced_eos_token_id=None,
                                       suppress_tokens=[0, 1, 2]).cpu().tolist()
            for index, tokens in zip(indices, generated):
                predictions[index] = tokenizer.decode(tokens, skip_special_tokens=True)
                no_eos += 3 not in tokens[1:]
    torch.cuda.synchronize()
    phases['generation_and_decode'] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    output = Path(args.output)
    output.write_text(''.join(json.dumps({'code': code}, ensure_ascii=False) + '\n' for code in predictions), encoding='utf-8', newline='\n')
    checked = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert len(checked) == 1919 and all(set(row) == {'code'} and isinstance(row['code'], str) for row in checked)
    syntax_error_rows = []
    for index, row in enumerate(checked, 1):
        try:
            ast.parse(row['code'])
        except SyntaxError:
            syntax_error_rows.append(index)
    torch.cuda.synchronize()
    phases['write_and_validate'] = time.perf_counter() - phase_started
    seconds = time.perf_counter() - begin
    result = {'gpu': gpu, 'parameters': parameters, 'rows': len(checked), 'cold_seconds': seconds,
              'max_new_tokens': settings['max_new_tokens'], 'batch_size': batch_size, 'no_eos': no_eos,
              'source_clipped': source_clipped, 'checkpoint_sha256': checkpoint_sha256,
              'actual_t4': actual_t4, 'within_180_seconds': seconds <= 180,
              'gpu_capability': list(capability), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
              'python_version': platform.python_version(), 'torch_version': torch.__version__,
              'transformers_version': transformers.__version__, 'cuda_version': torch.version.cuda,
              'phase_seconds': phases, 'syntax_error_rows': syntax_error_rows,
              'empty_code_rows': [i for i, row in enumerate(checked, 1) if not row['code'].strip()],
              'questions_lf_sha256': normalized_questions_sha256,
              'submission_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'questions_sha256': hashlib.sha256(Path(args.questions).read_bytes()).hexdigest()}
    output.with_suffix('.timing.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not args.allow_other_gpu and seconds > 180:
        raise RuntimeError('T4 180-second requirement failed')


if __name__ == '__main__':
    main()
