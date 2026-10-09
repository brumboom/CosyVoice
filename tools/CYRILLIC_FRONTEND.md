# Cyrillic frontend regression

The Chinese/English text-normalization backends do not support Cyrillic input.
Even without either optional backend, the fallback converts digit sequences to
English words: `У неё 12 яблок.` becomes `У неё twelve яблок.`. Mixed Chinese and
Cyrillic text can also enter the Chinese punctuation/whitespace rewrite path.

The patch routes text containing Unicode Cyrillic characters around these
language-specific transformations, while retaining sentence grouping. Numbers,
dates, `ё`, mixed-script content and punctuation reach the model unchanged.
The sentence token budget remains soft: long individual sentences are preserved,
not truncated. Short trailing chunks may be merged. `text_frontend=True` remains
the default and no inference signatures change. English and Chinese inputs
without Cyrillic retain their existing paths. This is not a general multilingual
normalizer or a Russian number/morphology ruleset.

## Tests without model weights

With the usual CosyVoice Python dependencies installed:

```sh
PYTHONPATH=.:third_party/Matcha-TTS python -m unittest discover -s tests -p test_cyrillic_frontend.py
```

The backend tests use doubles and check routing, not the linguistic behavior of
installed WeText or ttsfrd resources. The original fallback fails the Cyrillic
number-preservation test; the patched implementation passes. Tests also cover
mixed scripts, dates, punctuation, long text, backend dispatch and compatibility.

## Paired CPU reproduction

Keep a clean checkout of the baseline commit alongside the patched checkout.
Do not run a serving copy of the model at the same time. The script hides CUDA,
sets two CPU threads, and loads one model. Use a fresh output directory per run.

```sh
PYTHONPATH=.:third_party/Matcha-TTS python tools/reproduce_cyrillic_frontend.py \
  --model-dir /models/Fun-CosyVoice3-0.5B-2512 \
  --model-revision 29e01c4e8d000f4bcd70751be16fa94bf3d85a18 \
  --baseline-root /checkouts/CosyVoice-original \
  --reference /references/reference.mp3 --transcript /references/reference.txt \
  --output /results/cyrillic-01
```

Default: 12 phrases, seeds 42/43/44, zero-shot, three paired modes (original true,
original false, patched true): 108 WAV files. For the issue sentence across
inference methods, add `--indices 0 --methods zero_shot cross_lingual instruct2`.
CosyVoice3 cross-lingual requires the assistant/end-of-prompt prefix in the
target (see `example.py`). Its special-token handling already skips normalization
regardless of the frontend flag; these runs check inference compatibility, not
the new normalization branch. Omitting the prefix fails an assertion and is not
evidence of a language-switching bug.
The baseline and patched checkout must have identical existing helper functions;
the script loads the baseline normalization method against these shared helpers.

The output includes environment/source hashes, reference checksum, normalized
text, text/prompt tokens, hashes of input tensors, timing and WAV files. It also
copies/resamples the reference and stores its transcript: inspect outputs before
publishing them. Backend `none` means no optional normalizer was initialized,
not successful testing of WeText/ttsfrd. Source/model revisions should be pinned.
Listening status deliberately remains `pending`; ASR does not certify audio.

For a redistributable reference, the CC0 Common Voice mirror
[Sh1man/common_voice_21_ru](https://huggingface.co/datasets/Sh1man/common_voice_21_ru)
at revision `9868a070163572c0ebda83ebb96c7f56f8b56e18` contains these test samples:

| Sample | Transcript | Original MP3 SHA-256 |
| --- | --- | --- |
| common_voice_ru_38615634 | Никогда со мной этого не бывало. | ae61b9fc2ac27c6e4bd3820b09b852e856d1ee5ad7e789cab9cce77e2b504fd8 |
| common_voice_ru_30353798 | Никогда со мной этого не бывало. | 99c2c2a687cc9d545e13fe3df29a7df91a3cfe5f31f760703022b15e700c0295 |

They are separate recordings, not a verified pair of distinct speakers. Download
the test shard or use the Dataset Viewer rows endpoint (`config=default`,
`split=test`, offsets 0 and 1); check the checksum. Viewer asset URLs expire.
No personal voice recording is part of this change.

## Relationship to reported issues

Related: [#1790](https://github.com/QwenAudio/CosyVoice/issues/1790) and
[#1949](https://github.com/QwenAudio/CosyVoice/issues/1949).
The #1790 example contains no numbers. Incorrect English number expansion is a
confirmed separate defect, not proof of the reported switch to Chinese. Do not
claim that issue fixed unless its audio failure reproduces before the patch and
disappears afterwards under matched conditions. The original reporter's audio
is not available. German normalization from #1949 is outside this patch's scope.

## Initial validation (CPU)

- Base and current upstream `main`: `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`.
- Model revision: `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`.
- Torch/torchaudio 2.7.1+cpu, ONNX Runtime 1.22.0, Transformers 4.51.3;
  installed WeText 0.0.4 could not initialize its remote resources. Backend: none.
- Six regression tests pass after the patch. Running the same suite against the
  original normalization method produces 13 failing subtests/assertions.
- On one local reference, the #1790 sentence in zero-shot at seed 42 produces
  byte-identical WAVs for all three modes (4.16 seconds each). This does not
  reproduce the reported Chinese output and is not evidence that #1790 is fixed.
- The full public-reference matrix, real WeText/ttsfrd runs, GPU runs and human
  listening remain separate acceptance checks; routing doubles do not cover them.
