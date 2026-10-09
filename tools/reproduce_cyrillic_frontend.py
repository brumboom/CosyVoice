"""CPU-only paired frontend reproduction with one loaded CosyVoice3 model."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time
import types

# Set before importing torch / ONNX Runtime, including when CUDA is installed.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import numpy as np
import soundfile as sf
import torch

import cosyvoice.cli.frontend as frontend_module
from cosyvoice.cli.cosyvoice import AutoModel


PHRASES = [
    'Таким образом, мы услышим от них более сфокусированные и полезные выступления.',
    'Неожиданно катастрофа приобрела глобальные масштабы.',
    'Привет! Как дела?',
    'Ёж шёл вдоль реки и нашёл жёлтый лист.',
    'Сегодня 8 октября 2026 года.',
    'Стоимость заказа составляет 1250 рублей 50 копеек.',
    'Встреча начнётся в 15:30 и закончится через 45 минут.',
    'В МГУ открыли новую лабораторию, а результаты передали в РАН.',
    'На двери висел замок, а на холме стоял старинный замок.',
    'Пожалуйста, позвоните мне завтра и расскажите, как прошла встреча.',
    'Вы действительно готовы отправиться в путешествие? Отлично, тогда начинаем!',
    'Когда над городом взошло солнце, улицы постепенно наполнились людьми. '
    'Кто-то спешил на работу, а кто-то остановился у маленькой пекарни, чтобы купить свежий хлеб.',
]
PREFIX = 'You are a helpful assistant.<|endofprompt|>'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def original_normalize(root):
    """Load only the trusted baseline method; reuse unchanged helper functions."""
    source = (root / 'cosyvoice/cli/frontend.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CosyVoiceFrontEnd')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'text_normalize')
    namespace = vars(frontend_module).copy()
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(root), 'exec'), namespace)
    return namespace['text_normalize'], sha(source.encode())


def git_revision(root):
    result = subprocess.run(['git', '-c', f'safe.directory={root}', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True, help='Trusted original checkout matching the patched checkout base')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--indices', nargs='+', type=int, default=list(range(len(PHRASES))))
    parser.add_argument('--methods', nargs='+', choices=['zero_shot', 'cross_lingual', 'instruct2'], default=['zero_shot'])
    args = parser.parse_args()
    transcript = args.transcript.read_text(encoding='utf-8').strip()
    if not transcript:
        parser.error('The reference transcript must not be empty.')
    if any(index < 0 or index >= len(PHRASES) for index in args.indices):
        parser.error('Phrase indices must be between 0 and 11.')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError('This reproducer must not use CUDA.')
    baseline, baseline_hash = original_normalize(args.baseline_root)
    patched = frontend_module.CosyVoiceFrontEnd.text_normalize
    root = Path(frontend_module.__file__).resolve().parents[2]
    metadata = {
        'source_commit': git_revision(root), 'baseline_commit': git_revision(args.baseline_root),
        'baseline_frontend_sha256': baseline_hash,
        'patched_frontend_sha256': sha(Path(frontend_module.__file__).read_bytes()),
        'patched_utils_sha256': sha((root / 'cosyvoice/utils/frontend_utils.py').read_bytes()),
        'reproducer_sha256': sha(Path(__file__).read_bytes()),
        'model_revision': args.model_revision, 'reference_sha256': sha(args.reference.read_bytes()),
        'reference_text': transcript, 'device': 'cpu', 'threads': 2,
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchaudio', 'onnxruntime', 'wetext', 'transformers']},
    }
    reference = args.output / 'reference-24k.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(args.reference), '-ac', '1', '-ar', '24000', str(reference)], check=True)
    info = sf.info(reference)
    if not 0 < info.duration <= 30:
        raise ValueError('Reference audio must contain between 0 and 30 seconds of speech.')
    metadata['reference_duration'] = info.duration
    model = AutoModel(model_dir=str(args.model_dir), fp16=False, load_trt=False, load_vllm=False)
    frontend = model.frontend
    metadata['backend'] = frontend.text_frontend or 'none'
    (args.output / 'environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    trace = []
    original_zero = frontend.frontend_zero_shot

    def capture(*values, **kwargs):
        inputs = original_zero(*values, **kwargs)
        trace.append({key: {'shape': list(value.shape), 'sha256': sha(value.detach().cpu().numpy().tobytes())}
                      for key, value in inputs.items() if isinstance(value, torch.Tensor)})
        return inputs

    frontend.frontend_zero_shot = capture
    rows = []
    for index in args.indices:
        text = PHRASES[index]
        for method in args.methods:
            for seed in args.seeds:
                for mode, normalize, enabled in [('original-true', baseline, True), ('original-false', baseline, False), ('patched-true', patched, True)]:
                    frontend.text_normalize = types.MethodType(normalize, frontend)
                    instruction = 'You are a helpful assistant. Speak in Russian.<|endofprompt|>'
                    prompt = PREFIX + transcript if method == 'zero_shot' else instruction if method == 'instruct2' else ''
                    # CosyVoice3 cross-lingual has no prompt text; the required
                    # end-of-prompt token goes in the target, as in example.py.
                    synthesis_text = PREFIX + text if method == 'cross_lingual' else text
                    normalized = frontend.text_normalize(synthesis_text, text_frontend=enabled)
                    normalized_prompt = frontend.text_normalize(prompt, split=False, text_frontend=enabled)
                    row = {'index': index, 'text': text, 'synthesis_text': synthesis_text, 'method': method, 'seed': seed, 'mode': mode,
                           'normalization_skipped_for_special_tokens': method == 'cross_lingual',
                           'normalized_text': normalized, 'normalized_prompt': normalized_prompt,
                           'text_tokens': [frontend.tokenizer.encode(part, allowed_special='all') for part in normalized],
                           'prompt_tokens': frontend.tokenizer.encode(normalized_prompt, allowed_special='all'),
                           'listening_review': 'pending'}
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    trace.clear()
                    start = time.perf_counter()
                    if method == 'zero_shot':
                        generated = model.inference_zero_shot(text, prompt, str(reference), stream=False, text_frontend=enabled)
                    elif method == 'cross_lingual':
                        generated = model.inference_cross_lingual(synthesis_text, str(reference), stream=False, text_frontend=enabled)
                    else:
                        generated = model.inference_instruct2(text, instruction, str(reference), stream=False, text_frontend=enabled)
                    try:
                        chunks = [part['tts_speech'].detach().cpu() for part in generated]
                        if not chunks:
                            raise RuntimeError('Empty audio output')
                        audio = torch.cat(chunks, dim=-1).squeeze(0).numpy()
                        name = f'{index:02d}-{method}-{seed}-{mode}.wav'
                        sf.write(args.output / name, audio, model.sample_rate, subtype='PCM_16')
                        row.update(file=name, duration=len(audio) / model.sample_rate, sample_rate=model.sample_rate)
                    except Exception as exc:
                        row['error'] = repr(exc)
                    row.update(seconds=time.perf_counter() - start, model_input_trace=list(trace))
                    rows.append(row)
                    (args.output / 'manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(json.dumps({key: row[key] for key in ['index', 'method', 'seed', 'mode', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
