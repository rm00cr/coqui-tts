import tqdm
import os
import pandas as pd
import torch
import time
# Test XTTS anonymize_inference
import numpy as np
import json

from concurrent.futures import ThreadPoolExecutor

import warnings
import soundfile as sf

from model_conf import ModelPaths, load_tts_and_trainer
from development.utils import assess_segment_quality, denoised_temp_file, generate_text_segments_with_whisper, get_unique_temp_filename, iterative_segment_refinement, split_audio
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import fcntl
import time


# for speaker similarity
from huggingface_hub import hf_hub_download
import torchaudio

import threading

# ADD THIS at module level
checkpoint_lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"


model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(model_file, map_location=device)

# Word error rate and blue score
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothie = SmoothingFunction().method4

"""
Checkpointing functions
"""
# needed for checkpointing
def load_checkpoint(checkpoint_path):
    """Load existing checkpoint if it exists"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"completed": [], "metadata": []}

def save_checkpoint(checkpoint_path, completed_pairs, metadata):
    """Thread-safe checkpoint saving"""
    with checkpoint_lock:
        checkpoint = {
            "completed": completed_pairs,
            "metadata": metadata
        }
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

def save_checkpoint_with_file_lock(checkpoint_path, completed_pairs, metadata):
    """File-based locking prevents multiple processes from corrupting the checkpoint"""
    lock_file = checkpoint_path + ".lock"
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            # Try to acquire exclusive lock
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # We have the lock - safe to write
                checkpoint = {
                    "completed": completed_pairs,
                    "metadata": metadata
                }
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                
                print(f"Checkpoint saved successfully (attempt {attempt + 1})")
                break
                
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = 0.1 * (2 ** attempt)  # Exponential backoff: 0.1, 0.2, 0.4, 0.8...
                print(f"Lock busy, retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
            else:
                print(f"Failed to acquire lock after {max_retries} attempts: {e}")
                raise
    
    # Clean up lock file
    try:
        os.unlink(lock_file)
    except:
        pass

def is_pair_completed(ref_file, target_file, completed_pairs):
    """Check if this ref-target pair has already been processed"""
    pair_id = f"{ref_file}_{target_file}"
    return pair_id in completed_pairs


"""
caching functions
"""
speaker_embedding_cache = {}
def get_speaker_embedding(path):

    if isinstance(path, str) and path not in speaker_embedding_cache:
        path_name = path
        audio = resample_audio(path).to(device)
        speaker_embedding_cache[path] = ecapa2(audio)
    elif isinstance(path, list):
        path_name = path[0].split('/')[-2]
        if path_name not in speaker_embedding_cache:
            audio = resample_audio(path).to(device)
            with torch.no_grad():
                    embedding = ecapa2(audio)
                    speaker_embedding_cache[path_name] = embedding
            del audio
            if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        path_name = path
    return speaker_embedding_cache[path_name]

whisper_text_cache = {}
def get_whisper_text(audio_path, asr_model):
    if audio_path not in whisper_text_cache:
        whisper_text_cache[audio_path] = asr_model.transcribe(audio_path)["text"].lower()
    return whisper_text_cache[audio_path]

def resample_audio(input_path, target_sr=16000):
    assert type(input_path) in [str, list], "input_path must be a string or a list of strings"
    if isinstance(input_path, str):
        audio, sr = torchaudio.load(input_path)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
    else:
        long_audio = []
        for path in input_path:
            audio, sr = torchaudio.load(path)
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
            long_audio.append(audio)
        audio = torch.cat(long_audio, dim=1)
        max_total_samples = target_sr * 60  # 60 seconds total max
        if audio.shape[1] > max_total_samples:
            audio = audio[:, :max_total_samples]
    return audio

def calc_speaker_similarity(ref_audio_path, anonymized_audio_path):
    ref_embedding = get_speaker_embedding(ref_audio_path).to(device)
    anonymized_embedding = get_speaker_embedding(anonymized_audio_path).to(device)
    # Compute cosine similarity
    with torch.no_grad():
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

def segment_quality_score(asr_model, target_speaker_emb, ref_speaker_emb, anonymized_wav, ecapa_model):
    if isinstance(anonymized_wav, np.ndarray):
        anonymized_wav = torch.tensor(anonymized_wav)
    elif not isinstance(anonymized_wav, torch.Tensor):
        anonymized_wav = torch.tensor(np.array(anonymized_wav))
    
    segment_length = 2 * 24000
    segments = split_audio(anonymized_wav, segment_length)
    text_segments = generate_text_segments_with_whisper(asr_model, segments, 24000)
    temp_files_created = [] 
    quality_infos = {}
    quality_scores = []

    for i, (audio_seg, text_seg) in enumerate(zip(segments, text_segments)):
        segment_file = get_unique_temp_filename(f'_segment_{i}.wav')
        sf.write(segment_file, audio_seg.detach().cpu().numpy().squeeze(), 24000)
        temp_files_created.append(segment_file)

        quality_info = assess_segment_quality(
            audio_seg, text_seg, target_speaker_emb, ref_speaker_emb, asr_model, ecapa_model
        )
        quality_infos[i] = quality_info
        if quality_info is not None:
                # Combine WER and BLEU into a single quality score for this segment
                segment_score = (1 - quality_info['wer']) * 0.5 + quality_info['blue'] * 0.5
                quality_scores.append(segment_score)
    
    for temp_file in temp_files_created:
            try:
                os.unlink(temp_file)
            except:
                pass

    # Combine WER and BLEU into a single quality score
    quality_score = (1 - quality_info['wer']) * 0.5 + quality_info['blue'] * 0.5
    return quality_score, quality_infos


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_reference(ref_file, model, output_dir, train_model, config, model_name, tts, checkpoint_path):
    """
    alpha: float, between 0 and 1, controls the degree of anonymization. 1 full anonymization (take style embedding of reference voice), 0 means take the style embedding from the target (the input audio we want to anonymize)\n
    style: str, one of 'target', 'mixing', 'half_and_half'. Determines the style of voice conversion.\n
    """
    import whisper
    import soundfile as sf

    checkpoint = load_checkpoint(checkpoint_path)

    # get whisper
    asr_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
    
    ref_path = os.path.join(references_dir, ref_file)
    ref_speakers = os.listdir(ref_path)
    ref_lang = ref_file.split('_')[0]

    ref_speakers = [os.path.join(ref_path, spk) for spk in ref_speakers]
    local_metadata = []

    for target_speaker_folder in tqdm.tqdm(os.listdir(targets_dir), leave=False, desc=f"{ref_file} targets"):
        if is_pair_completed(ref_file, target_speaker_folder, checkpoint["completed"]):
            tqdm.tqdm.write(f"Skipping {ref_file} -> {target_speaker_folder} (already completed)")
            continue
        target_path = os.path.join(targets_dir, target_speaker_folder)
        target_speakers = os.listdir(target_path)
        target_speakers = [os.path.join(target_path, spk) for spk in target_speakers]
        
        anonymized_wav = None

        lan = target_speaker_folder.split('_')[0]
        if lan != ref_lang:
            continue  # skip if languages do not match
        """

        if target_speakers[0] in ref_speakers:
            continue  # skip if target is same as reference """

        if model_name == 'xtts2_forward_iteration':
            #with denoised_temp_file(target_speakers[0]) as denoised_path:
            result = model.forward_iteration(
                lang=lan.lower(),
                text=get_whisper_text(target_speakers[0], asr_model),
                target_sample=target_speakers[0],
                ref_sample=ref_speakers,
                train_model=train_model,
                max_conditioning_length=config.model_args.max_conditioning_length,
                min_conditioning_length=config.model_args.min_conditioning_length,
                tts=tts,
                asr_model=asr_model,
                ecapa=ecapa2,
            )
            # get best audio based on quality scores
            best = np.where(np.max(result[2])==result[2])[0][0]
            # save first audio 
            anonymized_wav = result[3][best]
        
        tqdm.tqdm.write(f"Anonymization done for {ref_file} + {target_speaker_folder}")
        _,segment_quality_score_value = segment_quality_score(
                asr_model,
                get_speaker_embedding(target_speakers[0]).to(device),
                get_speaker_embedding(ref_speakers).to(device),
                anonymized_wav,
                ecapa2
            )
        
        if isinstance(anonymized_wav, torch.Tensor):
            anonymized_wav = anonymized_wav.detach().cpu().numpy().squeeze()
        elif isinstance(anonymized_wav, list):
            anonymized_wav = np.array(anonymized_wav)
        

        #anonymized_wav = result["wav"]
        
        if model_name == 'xtts2_forward_iteration':
            out_name = f"{os.path.splitext(ref_file)[0]}_to_{os.path.splitext(target_speaker_folder)[0]}"
            # save first audio only
            for index, wav in enumerate(result[3]):
                sf.write(os.path.join(output_dir, f"{out_name}_{index}.wav"),  np.array(wav), model.config.audio.output_sample_rate)
                break

        out_name = f"{os.path.splitext(ref_file)[0]}_to_{os.path.splitext(target_speaker_folder)[0]}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, anonymized_wav, model.config.audio.output_sample_rate)
        tqdm.tqdm.write(f"Saved: {out_path}")

        speaker_similarity_an_ref = calc_speaker_similarity(ref_speakers, out_path)
        speaker_similarity_an_target = calc_speaker_similarity(target_speakers[0], out_path)
        wer_score, bleu_score = calc_asr_wer_blue(target_speakers[0], out_path, asr_model)
        tqdm.tqdm.write(f"Processed: {ref_file} + {target_speaker_folder} | Speaker Similarity: {speaker_similarity_an_ref:.4f} | WER: {wer_score} | BLEU: {bleu_score}")

        local_metadata.append({
            "ref_file": ref_file,
            "ref_path": ref_path,
            "target_file": target_speaker_folder,
            "target_path": target_path,
            "output_file": out_name,
            "output_path": out_path,
            "WER": wer_score,
            "BLEU": bleu_score,
            "speaker_similarity_an_ref": speaker_similarity_an_ref,
            "speaker_similarity_an_target": speaker_similarity_an_target,
            "language": lan,
            "segment_quality_score": segment_quality_score_value,
            "min_ref_seg": min([i['ref_sim'] for i in segment_quality_score_value.values()]),
            "min_tar_seg": max([i['target_sim'] for i in segment_quality_score_value.values()]),
            "orig_target_path": target_speakers[0],
        })
        # ADD THIS: Mark this pair as completed and save checkpoint
        pair_id = f"{ref_file}_{target_speaker_folder}"
        # dont just fail the whole process if file already open
        while True:
            try:
                checkpoint = load_checkpoint(checkpoint_path)  # Reload fresh state
                checkpoint["completed"].append(pair_id)
                checkpoint["metadata"].append(local_metadata[-1])  # Use append, not extend
                save_checkpoint_with_file_lock(checkpoint_path, checkpoint["completed"], checkpoint["metadata"])
                break  # Exit loop if successful
            except Exception as e:
                tqdm.tqdm.write(f"Error loading checkpoint: {e}")
                time.sleep(30)  # Wait before retrying
    return local_metadata

def main(model_name: str, output_dir: str = output_dir, name:str =f'{time.time()}'):
    checkpoint_filename = f"checkpoint_{model_name}_{name}.json"
    checkpoint_path = os.path.join(csv_output_dir, checkpoint_filename)

    checkpoint = load_checkpoint(checkpoint_path)
    #metadata = checkpoint["metadata"]  # Start with existing metadata

    output_dir = os.path.join(output_dir, f"{model_name}_{name}")
    os.makedirs(output_dir, exist_ok=True)
    paths = ModelPaths()
    tts, model, train_model, config = load_tts_and_trainer(paths)

    ref_files = os.listdir(references_dir)
    with ThreadPoolExecutor(max_workers=4) as executor:  
        futures = [
            executor.submit(process_reference, ref_file, model,  output_dir, train_model, config, model_name, tts, checkpoint_path)
            for ref_file in ref_files
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="References"):
            future.result()


    # Save metadata to a CSV
    final_checkpoint = load_checkpoint(checkpoint_path)
    df = pd.DataFrame(final_checkpoint["metadata"])
    df.to_csv(os.path.join(csv_output_dir, f"metadata_{model_name}_{name}.csv"), index=False)
    tqdm.tqdm.write(f"Metadata saved to metadata_{model_name}_{name}.csv")
    """if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file {checkpoint_filename} removed (run completed)")"""

if __name__ == "__main__":
    main("xtts2_forward_iteration", output_dir, name=f'top_references_by_reference')