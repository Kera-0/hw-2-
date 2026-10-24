"""Check saved artifacts without executing any generated answer programs."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_submission(path):
    answers = rows(path)
    assert len(answers) == 1919
    assert all(set(row) == {'code'} and isinstance(row['code'], str) for row in answers)
    syntax_errors = []
    for i, row in enumerate(answers, 1):
        try:
            ast.parse(row['code'])
        except SyntaxError as exc:
            syntax_errors.append({'row': i, 'message': exc.msg})
    return {'rows': len(answers), 'sha256': sha256(path),
            'empty_rows': [i for i, row in enumerate(answers, 1) if not row['code'].strip()],
            'syntax_errors': syntax_errors}


def main():
    questions = ROOT / 'questions_used.jsonl'
    assert len(rows(questions)) == 1919
    normalized = hashlib.sha256(questions.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    assert normalized == 'a3a128e5acf694f9b9a828dcc582bacb79367eecf11af2f6041c0c5f788e7003'
    with zipfile.ZipFile(ROOT / 'code_submission.zip') as archive:
        assert archive.testzip() is None
        count = 0
        for name in archive.namelist():
            if name.endswith('.py'):
                ast.parse(archive.read(name).decode('utf-8-sig'), filename=name)
                count += 1
        manifest = json.loads(archive.read('code/artifacts/files_sha256.json'))
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read('code/' + name)).hexdigest() == expected, name
    result = {'source_commit': '043d2689b658687e8dce7c8c53b83c99311b79b2',
              'questions': {'rows': 1919, 'sha256': sha256(questions), 'lf_sha256': normalized},
              'archive': {'python_files_parsed': count, 'manifest_entries_checked': len(manifest)},
              'saved_submission': inspect_submission(ROOT / 'submission_130m.jsonl'),
              't4_benchmark': {'status': 'pending'},
              'limitations': ['Original bot question order has not been independently verified.',
                              'Syntax validation does not establish functional correctness or official pass@1.']}
    folder = ROOT / 'results' / 't4'
    timing = folder / 'submission_130m.timing.json'
    if timing.exists():
        report = json.loads(timing.read_text(encoding='utf-8'))
        result['t4_benchmark'] = report
        if report.get('t4_limit_verified'):
            assert report['status'] == 'completed' and report['actual_t4'] is True
            assert report['cold_subprocess_seconds'] <= 180 and report['process_returncode'] == 0
            actual = folder / 'submission_130m.jsonl'
            assert sha256(actual) == report['submission_sha256']
            assert report['questions_lf_sha256'] == normalized
            assert report['checkpoint_sha256'] == 'adf2134a8c637a7b50b0fbee3370475e2e64c719eaf29fe29f73104f8582abf7'
            result['t4_submission'] = inspect_submission(actual)
            old, new = rows(ROOT / 'submission_130m.jsonl'), rows(actual)
            changed = [i for i, (a, b) in enumerate(zip(old, new), 1) if a != b]
            result['t4_submission']['different_from_saved_rows'] = changed
            result['t4_submission']['identical_to_saved_rows'] = len(old) - len(changed)
    (ROOT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
