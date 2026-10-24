"""Embed the reviewed inference sources and keep the submission archive in sync."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source(cell, text):
    cell['source'] = text.splitlines(keepends=True)


def main():
    path = ROOT / 'inference_kaggle_t4.ipynb'
    nb = json.loads(path.read_text(encoding='utf-8'))
    cells = nb['cells']
    source(cells[0], '''# Генерация и полный замер на одной T4 в Kaggle
Выберите GPU T4 x2 и Internet, затем Run All. Используется только GPU 0.
Веса из релиза v1.0 и 1919 вопросов из зафиксированного коммита GitHub
скачиваются автоматически до начала замера; проверяются SHA256.
Можно также прикрепить Dataset с checkpoint_130m.zip и questions_used.jsonl.
''')
    source(cells[2], '''from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
assert INPUT.is_dir() and WORK.is_dir(), 'Запустите ноутбук в Kaggle'
os.chdir(WORK)
hardware = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
    '--format=csv,noheader'], text=True)
print(hardware)
assert hardware.splitlines() and hardware.splitlines()[0].split(',')[1].strip() in {
    'Tesla T4', 'NVIDIA Tesla T4', 'NVIDIA T4', 'T4'}, 'Нужна GPU T4'
Path('hardware_preflight.txt').write_text(hardware, encoding='utf-8')

def find_unique(name):
    matches = sorted(INPUT.rglob(name))
    assert len(matches) <= 1, f'Найдено несколько {name}; оставьте один Dataset'
    return matches[0] if matches else None

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def download(url, destination, expected_sha256):
    if destination.exists() and sha256(destination) == expected_sha256:
        return
    temporary = destination.with_suffix('.download')
    print('Скачивание', destination.name, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
        shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
    assert sha256(temporary) == expected_sha256, f'Неверный SHA256: {destination.name}'
    temporary.replace(destination)

questions = find_unique('questions_used.jsonl')
if questions:
    shutil.copy2(questions, WORK / questions.name)
else:
    download('https://raw.githubusercontent.com/Kera-0/hw-2-/043d2689b658687e8dce7c8c53b83c99311b79b2/questions_used.jsonl',
             WORK / 'questions_used.jsonl',
             'a3a128e5acf694f9b9a828dcc582bacb79367eecf11af2f6041c0c5f788e7003')
archive = find_unique('checkpoint_130m.zip')
if archive:
    shutil.copy2(archive, WORK / archive.name)
else:
    weights = find_unique('model.safetensors')
    if weights:
        checkpoint = weights.parent
        names = ['config.json', 'generation_config.json', 'model.safetensors',
                 'tokenizer.json', 'training_meta.json', 'inference_config.json']
        assert all((checkpoint / name).is_file() for name in names), 'Неполный checkpoint'
        with zipfile.ZipFile(WORK / 'checkpoint_130m.zip', 'w', compression=zipfile.ZIP_STORED) as target:
            for name in names:
                target.write(checkpoint / name, 'best/' + name)
    else:
        download('https://github.com/Kera-0/hw-2-/releases/download/v1.0/checkpoint_130m.zip',
                 WORK / 'checkpoint_130m.zip',
                 'adf2134a8c637a7b50b0fbee3370475e2e64c719eaf29fe29f73104f8582abf7')
print('Входные файлы готовы:', WORK)
''')
    embedded = '''from pathlib import Path
import os
BASE = Path('/kaggle/working')
os.chdir(BASE)
QUESTIONS_FILE = 'questions_used.jsonl'
for name in ['checkpoint_130m.zip', QUESTIONS_FILE]:
    assert Path(name).is_file(), f'Отсутствует {name}'
'''
    for variable, filename in [('INFERENCE_PROGRAM', 'inference_130m.py'), ('BENCHMARK_PROGRAM', 'benchmark_130m.py')]:
        program = (ROOT / 'code' / filename).read_text(encoding='utf-8')
        embedded += f'{variable} = {program!r}\nPath({filename!r}).write_text({variable}, encoding="utf-8")\n'
    source(cells[3], embedded)
    source(cells[5], '''import subprocess, sys, json
BATCH_SIZE = 128
with zipfile.ZipFile('checkpoint_130m.zip') as archive:
    settings = json.loads(archive.read('best/inference_config.json'))
MAX_NEW_TOKENS = settings['max_new_tokens']
print('Лимит токенов:', MAX_NEW_TOKENS)
subprocess.run([sys.executable, '-u', 'benchmark_130m.py', '--questions', QUESTIONS_FILE,
                '--batch-size', str(BATCH_SIZE), '--max-new-tokens', str(MAX_NEW_TOKENS)],
               check=True, timeout=230)
report = json.loads(Path('submission_130m.timing.json').read_text(encoding='utf-8'))
assert report['rows'] == 1919 and report['parameters'] == 130183936
assert report['actual_t4'] and report['t4_limit_verified'] and report['status'] == 'completed'
print('Все 1919 строк на одной T4:', report['cold_subprocess_seconds'], 'секунд')
print('Незавершённых генераций:', report['no_eos'], 'синтаксические ошибки в строках:', report['syntax_error_rows'])
''')
    source(cells[7], '''from IPython.display import FileLink, display
for name in ['submission_130m.jsonl', 'submission_130m.timing.json', 'submission_130m.log', 'hardware_preflight.txt']:
    display(FileLink(name))
''')
    for index, c in enumerate(cells):
        c['id'] = f'cell-{index:02d}'
        if c['cell_type'] == 'code':
            c.update(execution_count=None, outputs=[])
    nb['metadata']['kaggle'] = {'accelerator': 'gpu', 'isInternetEnabled': True, 'isGpuEnabled': True}
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    # The historical Colab notebook uses the same programs, retaining its own setup.
    archive_path = ROOT / 'code_submission.zip'
    with zipfile.ZipFile(archive_path) as old:
        entries = [(info, old.read(info.filename)) for info in old.infolist()]
    updates = {'code/' + name: (ROOT / 'code' / name).read_bytes()
               for name in ['inference_130m.py', 'benchmark_130m.py']}
    colab = json.loads(dict((info.filename, data) for info, data in entries)['code/inference_t4.ipynb'])
    for index, cell in enumerate(colab['cells']):
        cell['id'] = f'cell-{index:02d}'
        text = ''.join(cell['source'])
        if 'INFERENCE_PROGRAM =' in text:
            prefix = text[:text.index('INFERENCE_PROGRAM =')]
            source(cell, prefix + embedded[embedded.index('INFERENCE_PROGRAM ='):])
    updates['code/inference_t4.ipynb'] = (json.dumps(colab, ensure_ascii=False, indent=1) + '\n').encode()
    manifest = json.loads(dict((i.filename, data) for i, data in entries)['code/artifacts/files_sha256.json'])
    for name, data in updates.items():
        manifest[name.removeprefix('code/')] = hashlib.sha256(data).hexdigest()
    updates['code/artifacts/files_sha256.json'] = json.dumps(manifest, indent=2).encode()
    temporary = archive_path.with_suffix('.download')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as target:
        for info, data in entries:
            target.writestr(info, updates.get(info.filename, data))
    temporary.replace(archive_path)


if __name__ == '__main__':
    main()
