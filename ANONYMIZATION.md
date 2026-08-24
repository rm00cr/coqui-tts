# Speaker anonymization with XTTS v2

Removes the speaker's identity from a recording while keeping what they said.

You give it two things:

| | | |
|---|---|---|
| **target** | the audio you want to anonymize | its **words** are preserved |
| **reference** | a donor voice | its **voice** is what you hear in the output |

The model resynthesizes the target's speech in the donor's voice. The transcript
survives; the original speaker's vocal identity does not.

> **The transcript is preserved verbatim.** This hides *who* is speaking, not *what* was
> said. If the words themselves identify someone — names, addresses, case numbers — you
> need text redaction as well. See [Limitations](#limitations).

---

## Install

```bash
git clone <this-repo> && cd coqui-tts
uv sync                       # or: pip install -e .
anonymize download-model      # ~2 GB of XTTS v2 checkpoints, once
```

`download-model` writes to `./XTTS_v2.0_original_model_files` by default. Put them
somewhere else and point at them with `--model-dir`, or:

```bash
export XTTS_MODEL_DIR=/path/to/checkpoints
```

A CUDA GPU is strongly recommended. CPU works for `single` mode but is slow, and the
`refine` and `iterate` modes are impractical without a GPU.

## Quickstart

### Command line

```bash
# one file
anonymize run interview.wav --reference donor.wav -o interview_anon.wav

# report how well it worked
anonymize run interview.wav --reference donor.wav -o out.wav --score

# a whole folder, resumable
anonymize batch recordings/ --reference donor.wav -o anonymized/ --resume

# German input, higher quality
anonymize run aufnahme.wav --reference donor.wav -o out.wav --lang de --mode iterate
```

### Python / Jupyter

```python
from anonymizer import Anonymizer

anon = Anonymizer()                                    # nothing loaded yet
result = anon.anonymize("interview.wav", reference="donor.wav")

anon.play(result)                 # audio widget in a notebook
anon.save(result, "out.wav")
print(result.text)                # the transcript it worked from
```

The checkpoints load on the first `anonymize` call and stay loaded, so keep one
`Anonymizer` around when processing several files.

```python
report = anon.score(result, "interview.wav")
print(report.to_dict())
# {'wer': 0.0, 'bleu': 1.0, 'target_similarity': 0.19,
#  'reference_similarity': 0.26, 'overall_quality': 0.49}
```

## Choosing the donor voice

`--reference` is required. It accepts:

```bash
--reference donor.wav                  # one file
--reference voices/                    # a directory; conditioning is averaged
--reference a.wav,b.wav,c.wav          # an explicit list
```

Several recordings of the same donor generally give a more stable voice than one. Aim
for clean speech of at least 6 seconds — the model slices a 3-6 second conditioning
window out of it.

To avoid passing `--reference` every time, set a default in your config:

```yaml
reference: ./voices/donor_01.wav
# or, to draw on a whole pool:
voice_pool_dir: ./voices
```

## Modes

| mode | what it does | cost | when |
|---|---|---|---|
| `single` | one forward pass | seconds | the default; bulk work |
| `refine` | segments the output, rescores each, regenerates the weak ones, crossfade-stitches | minutes, GPU | one-off files where quality matters |
| `iterate` | `iterations` full passes, each fed the previous output; keeps the best-scoring pass | minutes, GPU | pushing identity furthest from the original |

`iterate` moves the voice furthest from the original because each pass compounds the
conversion — but it also compounds transcription drift, so watch WER. On the bundled
sample, three iterations moved similarity-to-original from 0.198 to 0.141 while WER rose
from 0.00 to 0.07.

## Configuration

Copy `anonymizer.example.yaml`, edit, and pass it with `--config`. Every option is
documented in that file. Precedence, highest first:

```
CLI flag  >  --config file  >  environment variable  >  built-in default
```

Environment variables: `XTTS_MODEL_DIR`, `ANONYMIZER_DEVICE`, `ANONYMIZER_MODE`,
`ANONYMIZER_WHISPER_MODEL`, `ANONYMIZER_VOICE_POOL`.

Check what you actually resolved:

```bash
anonymize config --show --config my.yaml --mode refine
```

## Reading the scores

`--score` and `Anonymizer.score()` report four numbers:

| metric | meaning | want |
|---|---|---|
| `wer` | word error rate of the output against the input transcript | **low** — the words survived |
| `bleu` | sentence BLEU against the input transcript | **high** — same as above |
| `target_similarity` | speaker similarity to the **original** speaker | **low** — identity removed |
| `reference_similarity` | speaker similarity to the **donor** | **high** — identity replaced |

`overall_quality` combines them, weighted `0.05 / 0.05 / 0.30 / 0.60`. Speaker
similarity comes from [ECAPA2](https://huggingface.co/Jenthe/ECAPA2), downloaded on
first use.

The absolute similarity numbers are small even when anonymization works well — judge
them relative to each other, not against 1.0. What matters is that
`reference_similarity > target_similarity`.

## Batch processing

```bash
anonymize batch recordings/ --reference donor.wav -o anonymized/ --resume
```

Writes one `<name>_anonymized.wav` per input, plus `manifest.csv` recording the input,
output, mode and transcript for each. Progress is checkpointed to
`anonymize_checkpoint.json` under the output directory; `--resume` skips what is already
done. The checkpoint is written under a file lock, so several processes can share one
output directory.

A file that fails does not stop the batch — failures are listed at the end and the exit
code is non-zero.

## Limitations

- **Content is not redacted.** Names, addresses and other identifying *words* pass
  through untouched. Anonymizing identity in the audio is not the same as anonymizing
  the transcript.
- **Set the language.** It defaults to English. Pointing it at German audio without
  `--lang de` produces a garbled transcript and therefore garbled output.
- **Quality depends on the donor.** Noisy, short, or very atypical reference audio gives
  a weak voice. Use several clean clips of one speaker.
- **Transcription errors propagate.** The pipeline resynthesizes from a Whisper
  transcript, so whatever Whisper mishears is what gets spoken. Pass `--text` when you
  have a known-good transcript, or use `--whisper-model large-v3`.
- **Prosody and timing shift.** This is resynthesis, not a filter. Pauses, emphasis and
  rate come out different from the original.
- **Not a formal privacy guarantee.** These are empirical similarity scores against one
  speaker-verification model, not a proof that re-identification is impossible. Evaluate
  against your own threat model before relying on it.

## How it works

1. Transcribe the target with Whisper (skipped when you pass `--text`).
2. Tokenize the transcript and encode the target audio to GPT audio codes.
3. Derive conditioning latents and a speaker embedding from the **reference** audio.
4. Run the XTTS GPT with the target's content codes and the reference's conditioning.
5. Decode with HiFi-GAN at 24 kHz.
6. In `refine` / `iterate`, score the result with Whisper + ECAPA2 and regenerate.

The anonymization methods live on the `Xtts` class in
[TTS/tts/models/xtts.py](TTS/tts/models/xtts.py) (`forward_from_audios_and_text`,
`prep_batch`, `forward`, `forward_iteration`). Segment-level refinement is in
[development/utils.py](development/utils.py). The `anonymizer/` package is a façade over
these; the research entry points (`run_test.py`, `run_test_hdf5.py`) call the same code
directly and are unaffected by it.

## Package layout

```
anonymizer/
  pipeline.py    Anonymizer — the public API
  config.py      AnonymizerConfig and its precedence rules
  session.py     lazily-loaded XTTS / Whisper / ECAPA2
  modes.py       single | refine | iterate
  voices.py      donor voice resolution
  scoring.py     WER, BLEU, speaker similarity, quality score
  batch.py       folder processing with resumable checkpoints
  audio.py       load / save / denoise helpers
  cli.py         the `anonymize` command
  download.py    checkpoint fetching
  model_setup.py XTTS + GPTTrainer construction
```

## Tests

```bash
pytest tests/anonymizer_tests -v                  # no model weights needed
XTTS_MODEL_DIR=./XTTS_v2.0_original_model_files \
  pytest tests/anonymizer_tests/test_end_to_end.py -v    # real inference
```

The end-to-end tests skip themselves when the checkpoints are absent.
