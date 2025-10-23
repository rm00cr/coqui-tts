import tqdm
import os
import pandas as pd
import torch
import time
# Test XTTS anonymize_inference
import numpy as np

from concurrent.futures import ThreadPoolExecutor

import warnings

from model_conf import ModelPaths, load_tts_and_trainer
from development.utils import denoised_temp_file, iterative_segment_refinement
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")



# Paths
references_dir = "/home/romolo/VT1/coqui-tts/test_data/Dataset/references"
targets_dir = "/home/romolo/VT1/coqui-tts/test_data/Dataset/targets"
output_dir = "/home/romolo/VT1/prog/streamlit_prog/outputs/"
csv_output_dir = "/home/romolo/VT1/prog/streamlit_prog/outputs/csv/"


# Load your XTTS model (customize as needed)



# for speaker similarity
from huggingface_hub import hf_hub_download
import torchaudio


device = "cuda" if torch.cuda.is_available() else "cpu"
# automatically checks for cached file, optionally set `cache_dir` location
model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(model_file, map_location=device)

# Word error rate and blue score
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothie = SmoothingFunction().method4


speaker_embedding_cache = {}
def get_speaker_embedding(path):
    if path not in speaker_embedding_cache:
        audio = resample_audio(path).to(device)
        speaker_embedding_cache[path] = ecapa2(audio)
    return speaker_embedding_cache[path]

whisper_text_cache = {}
def get_whisper_text(audio_path, asr_model):
    if audio_path not in whisper_text_cache:
        whisper_text_cache[audio_path] = asr_model.transcribe(audio_path)["text"].lower()
    return whisper_text_cache[audio_path]

def resample_audio(input_path, target_sr=16000):
    audio, sr = torchaudio.load(input_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio

def calc_speaker_similarity(ref_audio_path, anonymized_audio_path):
    ref_audio = resample_audio(ref_audio_path)
    anonymized_audio = resample_audio(anonymized_audio_path)

    # Move audio to the same device as the model
    ref_audio = ref_audio.to(device)
    anonymized_audio = anonymized_audio.to(device)

    ref_embedding = ecapa2(ref_audio)
    anonymized_embedding = ecapa2(anonymized_audio)

    # Compute cosine similarity
    similarity = torch.cosine_similarity(ref_embedding, anonymized_embedding)
    return similarity.item()

def calc_asr_wer_blue(target_audio_path, anonymized_audio_path, asr_model):
    if asr_model is None:
        tqdm.write('ASR model not available')
        return None, None

    else:
        # Use Whisper ASR output for both audios
        ref_transcription = get_whisper_text(target_audio_path, asr_model)
        anon_transcription = get_whisper_text(anonymized_audio_path, asr_model)

        bleu_score = sentence_bleu([ref_transcription.split()], anon_transcription.split(), smoothing_function=smoothie)
        wer_score = wer(ref_transcription, anon_transcription)

        return wer_score, bleu_score


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_reference(ref_file, model, output_dir, train_model, config, model_name, tts):
    """
    alpha: float, between 0 and 1, controls the degree of anonymization. 1 full anonymization (take style embedding of reference voice), 0 means take the style embedding from the target (the input audio we want to anonymize)\n
    style: str, one of 'target', 'mixing', 'half_and_half'. Determines the style of voice conversion.\n
    """
    import whisper
    import soundfile as sf

    # get whisper
    asr_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    ref_path = os.path.join(references_dir, ref_file)
    ref_speakers = os.listdir(ref_path)
    ref_speakers = [os.path.join(ref_path, spk) for spk in ref_speakers]
    local_metadata = []

    for target_file in tqdm.tqdm(os.listdir(targets_dir)[:2], leave=False, desc=f"{ref_file} targets"):
        target_path = os.path.join(targets_dir, target_file)
        target_speakers = os.listdir(target_path)
        target_speakers = [os.path.join(target_path, spk) for spk in target_speakers]
        anonymized_wav = None
        if model_name == 'xtts2':
            result = model.forward_from_audios_and_text(
                'en',
                text=get_whisper_text(target_speakers[0], asr_model),
                target_sample=target_speakers[0],
                ref_sample=ref_speakers,
                train_model=train_model,
                max_conditioning_length=config.model_args.max_conditioning_length,
                min_conditioning_length=config.model_args.min_conditioning_length
            )
            anonymized_wav = result["wav"]
        elif model_name == 'xtts2_segment_refinement_overlap_th_0_6':
            anonymized_wav = iterative_segment_refinement(
                model=model,                                    # Your XTTS model
                target_speaker=target_speakers[0],              # Path to target speaker audio
                ref_speaker=ref_speakers,                        # List of reference samples
                max_conditioning_length=config.model_args.max_conditioning_length,
                min_conditioning_length=config.model_args.min_conditioning_length,
                train_model=train_model,                        # Your trained model
                asr_model=asr_model,                           # Your ASR model
                ecapa_model=ecapa2,                            # Your ECAPA model
                lang='en',                                     # Language
                threshold=0.6,                                 # Quality threshold (adjust as needed)
                max_attempts=10
            )
        elif model_name == 'xtts2_forward_iteration_denoise':
            with denoised_temp_file(target_speakers[0]) as denoised_path:
                result = model.forward_iteration(
                    lang='en',
                    text=get_whisper_text(target_speakers[0], asr_model),
                    target_sample=denoised_path,
                    ref_sample=ref_speakers[0],
                    train_model=train_model,
                    max_conditioning_length=config.model_args.max_conditioning_length,
                    min_conditioning_length=config.model_args.min_conditioning_length,
                    tts=tts,
                    asr_model=asr_model,
                    ecapa=ecapa2,
                )
            # get best audio based on quality scores
            best = np.where(np.max(result[2])==result[2])[0][0]
            anonymized_wav = result[3][best]

        if isinstance(anonymized_wav, torch.Tensor):
            anonymized_wav = anonymized_wav.detach().cpu().numpy().squeeze()
        elif isinstance(anonymized_wav, list):
            anonymized_wav = np.array(anonymized_wav)
        

        #anonymized_wav = result["wav"]
        out_name = f"{os.path.splitext(ref_file)[0]}_to_{os.path.splitext(target_file)[0]}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, anonymized_wav, model.config.audio.output_sample_rate)
        tqdm.tqdm.write(f"Saved: {out_path}")

        speaker_similarity_an_ref = calc_speaker_similarity(ref_speakers[0], out_path)
        speaker_similarity_an_target = calc_speaker_similarity(target_speakers[0], out_path)
        wer_score, bleu_score = calc_asr_wer_blue(target_speakers[0], out_path, asr_model)
        tqdm.tqdm.write(f"Processed: {ref_file} + {target_file} | Speaker Similarity: {speaker_similarity_an_ref:.4f} | WER: {wer_score} | BLEU: {bleu_score}")

        local_metadata.append({
            "ref_file": ref_file,
            "ref_path": ref_path,
            "target_file": target_file,
            "target_path": target_path,
            "output_file": out_name,
            "output_path": out_path,
            "WER": wer_score,
            "BLEU": bleu_score,
            "speaker_similarity_an_ref": speaker_similarity_an_ref,
            "speaker_similarity_an_target": speaker_similarity_an_target,
            "language": "en"
        })
    return local_metadata

def main(model_name: str, output_dir: str = output_dir, name:str =f'{time.time()}'):
    metadata = []
    output_dir = os.path.join(output_dir, f"{model_name}_{name}")
    os.makedirs(output_dir, exist_ok=True)
    paths = ModelPaths()
    tts, model, train_model, config = load_tts_and_trainer(paths)

    ref_files = os.listdir(references_dir)
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = [
            executor.submit(process_reference, ref_file, model,  output_dir, train_model, config, model_name, tts)
            for ref_file in ref_files
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="References"):
            metadata.extend(future.result())

    # Save metadata to a CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(csv_output_dir, f"metadata_{model_name}_{name}.csv"), index=False)
    tqdm.tqdm.write(f"Metadata saved to metadata_{model_name}_{name}.csv")

if __name__ == "__main__":
    main("xtts2_forward_iteration_denoise", output_dir, name=f'{time.time()}_target_as_target_ref_as_ref_en')