import argparse
import h5py
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


from model_conf import ModelPaths, load_tts_and_trainer

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import fcntl
import time
import librosa
import argparse


# for speaker similarity
from huggingface_hub import hf_hub_download
import torchaudio

import threading


checkpoint_lock = threading.Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"


model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(model_file, map_location=device)

# Word error rate and blue score
from jiwer import wer
from sacrebleu import BLEU
bleu_scorer = BLEU(effective_order=True)

import re
import unicodedata


def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Replace German ß with ss
    text = text.replace('ß', 'ss')

    # Replace common Unicode variations (accents, umlauts, etc.)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    # Remove punctuation and keep only word characters and whitespace
    cleaned = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned


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
    lock_handle = None
    try:
        for attempt in range(max_retries):
            try:
                # Try to acquire exclusive lock
                lock_handle = open(lock_file, 'w')
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
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
                if lock_handle:
                    lock_handle.close()
                    lock_handle = None
                if attempt < max_retries - 1:
                    wait_time = 0.1 * (2 ** attempt)  
                    print(f"Lock busy, retrying in {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to acquire lock after {max_retries} attempts: {e}")
                    raise
    finally:
        # Clean up lock file properly
        if lock_handle:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            except:
                pass
        try:
            os.unlink(lock_file)
        except:
            pass

def is_pair_completed(ref_file, target_file, completed_pairs):
    """Check if this ref-target pair has already been processed"""
    pair_id = f"{ref_file}_{target_file}"
    return pair_id in completed_pairs



def get_speaker_embedding(audio_pointer,  orig_sr=22050,hdf5_path = None):
    speaker_embedding = None
    if isinstance(audio_pointer, str):
        path_name = audio_pointer
        audio = resample_audio_hdf5(audio_pointer, hdf5_path,target_sr=16000, orig_sr=orig_sr).to(device) if hdf5_path is not None else resample_audio(audio_pointer, target_sr=orig_sr).to(device)
        speaker_embedding = ecapa2(audio)
    elif isinstance(audio_pointer, list):
        audio = resample_audio_hdf5(audio_pointer, hdf5_path,target_sr=16000, orig_sr=orig_sr).to(device) if hdf5_path is not None else resample_audio(audio_pointer, target_sr=orig_sr).to(device)
        with torch.no_grad():
                speaker_embedding = ecapa2(audio)
        del audio
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise TypeError(f"audio_pointer must be string or list, got {type(audio_pointer)}")
    return speaker_embedding


def resample_audio_hdf5(store_id, hdf5_path, target_sr=16000,orig_sr=24000):
    """Load and resample audio from HDF5 file"""
    import h5py
    
    assert isinstance(store_id, (str, int, list)), "store_id must be string, int, or list"
    
    if isinstance(store_id, list):
        long_audio = []
        for sid in store_id:
            with h5py.File(hdf5_path, 'r') as f:
                audio_data = f[str(sid)][()]
            audio = torch.tensor(audio_data).unsqueeze(0).float()
            if audio.shape[0] == 1 and len(audio.shape) == 2:
                pass 
            elif len(audio.shape) == 1:
                audio = audio.unsqueeze(0)
            

            if audio.shape[1] > 0:  
                audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=target_sr)
            long_audio.append(audio)
        
        audio = torch.cat(long_audio, dim=1)
        max_total_samples = target_sr * 60
        if audio.shape[1] > max_total_samples:
            audio = audio[:, :max_total_samples]
    else:
        with h5py.File(hdf5_path, 'r') as f:
            audio_data = f[str(store_id)][()]
        audio = torch.tensor(audio_data).unsqueeze(0).float()
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=target_sr)
    
    return audio

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
        max_total_samples = target_sr * 60  
        if audio.shape[1] > max_total_samples:
            audio = audio[:, :max_total_samples]
    return audio

def calc_speaker_similarity(ref_audio_path, ref_sr, anonymized_audio_array,ref_path):
    ref_embedding = get_speaker_embedding(ref_audio_path, ref_sr,hdf5_path=ref_path).to(device)
    if isinstance(anonymized_audio_array, np.ndarray):
        anonymized_audio_tensor = torch.tensor(anonymized_audio_array).unsqueeze(0).float()
    elif isinstance(anonymized_audio_array, torch.Tensor):
        if len(anonymized_audio_array.shape) == 1:
            anonymized_audio_tensor = anonymized_audio_array.unsqueeze(0).float()
        else:
            anonymized_audio_tensor = anonymized_audio_array.float()
    else:
        raise TypeError(f"Expected numpy array or tensor, got {type(anonymized_audio_array)}")
    
    with torch.no_grad():
        anonymized_embedding = ecapa2(anonymized_audio_tensor.to(device)) #get_speaker_embedding(anonymized_audio_path).to(device)
    # Compute cosine similarity
    with torch.no_grad():
        similarity = torch.cosine_similarity(ref_embedding, anonymized_embedding,dim=1)
        return similarity.item()


from concurrent.futures import ThreadPoolExecutor, as_completed

embedding_cache_ref = {}

def process_reference(ref_speaker_id, model, output_dir, train_model, config, model_name, tts, checkpoint_path, df_ref, df_target, ref_hdf5_path, target_hdf5_path):
    """
    alpha: float, between 0 and 1, controls the degree of anonymization. 1 full anonymization (take style embedding of reference voice), 0 means take the style embedding from the target (the input audio we want to anonymize)\n
    style: str, one of 'target', 'mixing', 'half_and_half'. Determines the style of voice conversion.\n
    """
    import whisper

    while True:
        try:
            checkpoint = load_checkpoint(checkpoint_path)  # Reload fresh state
            break  # Exit loop if successful
        except Exception as e:
            tqdm.tqdm.write(f"Error loading checkpoint: {e}")
            time.sleep(30)  # Wait before retrying

    # get whisper
    asr_model = whisper.load_model("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
    
    
    ref_lang = df_ref.lang[df_ref.speaker_id == ref_speaker_id].iloc[0]


    local_metadata = []

    # get all speakers for the targets
    target_speaker_ids = df_target.speaker_id.unique()

    for target_speaker_id in tqdm.tqdm(target_speaker_ids, desc=f"{ref_speaker_id} targets"):
        store_ids = df_target.store_id[df_target.speaker_id == target_speaker_id].tolist()
        if "predicted_gender" in df_target.columns:
            target_gender = df_target.predicted_gender[df_target.speaker_id == target_speaker_id].iloc[0]
        elif "trial_file" in df_target.columns:
            trial_rows = df_target[df_target.speaker_id == target_speaker_id]
            if not trial_rows.empty:
                trial_file = trial_rows.trial_file.iloc[0]
                if "trials_f" in trial_file:
                    target_gender = 'female'
                elif "trials_m" in trial_file:
                    target_gender = 'male'
        for store_id in store_ids:
            if is_pair_completed(str(ref_speaker_id), f"{str(target_speaker_id)}_{store_id}", checkpoint["completed"]):
                tqdm.tqdm.write(f"Skipping {int(ref_speaker_id)} -> {str(target_speaker_id)}_{store_id} (already completed)")
                continue

            ref_store_ids = df_ref.store_id[(df_ref.speaker_id == ref_speaker_id) & (df_ref.predicted_gender == target_gender)].tolist()
            target_store_ids = store_id


            target_to_ref = []
            ref_similarities = []
            loaded_target_audio = resample_audio_hdf5(target_store_ids, target_hdf5_path,target_sr=16000, orig_sr=22050).squeeze().unsqueeze(0).to(device)
            tar_embedding = ecapa2(loaded_target_audio)
            for ref_id in ref_store_ids:
                # load ref_id audio and compute similarity to target
                if ref_id not in embedding_cache_ref:
                    loaded_ref_audio = resample_audio_hdf5(ref_id, ref_hdf5_path,target_sr=16000, orig_sr=22050).squeeze().unsqueeze(0).to(device)
                    ref_embedding = ecapa2(loaded_ref_audio)
                    embedding_cache_ref[ref_id] = ref_embedding
                else:
                    ref_embedding = embedding_cache_ref[ref_id]
                similarity = torch.cosine_similarity(ref_embedding, tar_embedding,dim=1)
                target_to_ref.append(similarity.item())
                ref_similarities.append((ref_id, similarity.item()))
            
            ref_similarities.sort(key=lambda x: x[1])
            target_to_ref = [sim for ref_id, sim in ref_similarities[:10]]
            ref_store_ids = [ref_id for ref_id, sim in ref_similarities[:10]]
            anonymized_wav = None

            lan = df_target.lang[df_target.store_id == store_id].iloc[0]
            if lan != ref_lang:
                continue  # skip if languages do not match

            if target_speaker_id == ref_speaker_id:
                continue  # skip if target is same as reference 
            
            text_row = df_target[df_target.store_id == store_id].text.iloc[0]
            predicted_text = df_target[df_target.store_id == store_id].transcript.iloc[0]
            
            # Truncate text to character limit based on language to prevent CUDA indexing errors
            char_limits = {
                "en": 250, "es": 239, "fr": 273, "de": 253, "it": 213,
                "pt": 203,  "nl": 251,
            }
            lang_code = lan.split("-")[0] if isinstance(lan, str) else lan
            char_limit = char_limits.get(lang_code, 250)

            if len(predicted_text) > char_limit:
                predicted_text = predicted_text[:char_limit]
                tqdm.tqdm.write(f"⚠️ Truncated text from {len(df_target[df_target.store_id == store_id].transcript.iloc[0])} to {char_limit} chars for language '{lang_code}'")
    

            if model_name == 'xtts2_forward_iteration':
                #with denoised_temp_file(target_speakers[0]) as denoised_path:
                result = model.forward_iteration_hdf5(
                    lang=lan.lower(),
                    text=predicted_text,
                    target_sample=target_store_ids,
                    ref_sample=ref_store_ids,
                    train_model=train_model,
                    max_conditioning_length=config.model_args.max_conditioning_length,
                    min_conditioning_length=config.model_args.min_conditioning_length,
                    tts=tts,
                    asr_model=asr_model,
                    ecapa=ecapa2,
                    target_hdf5_path=target_hdf5_path,  # ADD THIS
                    ref_hdf5_path=ref_hdf5_path,     # ADD THIS
                    n=5 
                )
                # get best audio based on quality scores
                best = np.where(np.max(result[2])==result[2])[0][0]
                # save first audio 
                anonymized_wav = result[3][best]
            
            tqdm.tqdm.write(f"Anonymization done for {ref_speaker_id} + {target_speaker_id}")
            
            if isinstance(anonymized_wav, torch.Tensor):
                anonymized_wav = anonymized_wav.detach().cpu().numpy().squeeze()
            elif isinstance(anonymized_wav, list):
                anonymized_wav = np.array(anonymized_wav)
        
            out_name = f"{ref_speaker_id}_to_{target_speaker_id}_{target_store_ids}"
            iteration_hdf5_path = os.path.join(output_dir, f"outputs.hdf5")
            key = f"{ref_speaker_id}_to_{target_speaker_id}_{target_store_ids}.wav"
            # save best audio
            try:
                with h5py.File(iteration_hdf5_path, 'a') as hdf5_iter:
                    if key in hdf5_iter:
                        del hdf5_iter[key]
                    hdf5_iter.create_dataset(key, data=anonymized_wav)
                    hdf5_iter[key].attrs['sr'] = model.config.audio.output_sample_rate
                    hdf5_iter[key].attrs['ref_speaker_id'] = ref_speaker_id
                    hdf5_iter[key].attrs['target_speaker_id'] = target_speaker_id
                    tqdm.tqdm.write(f"Saved output to HDF5: {iteration_hdf5_path}")
            except Exception as e:
                tqdm.tqdm.write(f"Error saving to HDF5: {e}")


            anonymized_wav_res = librosa.resample(anonymized_wav, orig_sr=24000, target_sr=16000)
            
            speaker_similarity_an_ref = calc_speaker_similarity(ref_store_ids, 22050, anonymized_wav_res, ref_hdf5_path)
            # calc target similarity
            anonymized_audio_tensor = torch.tensor(anonymized_wav_res).unsqueeze(0).float().to(device)
            anonymized_embedding = ecapa2(anonymized_audio_tensor)
            speaker_similarity_an_target = torch.cosine_similarity(anonymized_embedding, tar_embedding,dim=1)
            speaker_similarity_an_target = speaker_similarity_an_target.item()

            #speaker_similarity_an_target = calc_speaker_similarity([target_store_ids], 22050, anonymized_wav_res, target_hdf5_path)
            tqdm.tqdm.write(f"Speaker Similarity (Anonymized vs Ref): {speaker_similarity_an_ref:.4f}")
            tqdm.tqdm.write(f"Speaker Similarity (Anonymized vs Target): {speaker_similarity_an_target:.4f}")

            
            
            # calc WER and BLEU
            for i in range(3):
                try:
                    generated_text = asr_model.transcribe(anonymized_wav_res,language=lan)['text']
                    if generated_text is not None:
                        break
                except Exception as e:
                    tqdm.tqdm.write(f"Error in ASR transcription: {e}")
                    generated_text = ""
                    time.sleep(1)  # wait before retrying
            
            text_row = clean_text(text_row)
            generated_text = clean_text(generated_text)

            try:
                wer_score = wer(text_row, generated_text)
            except Exception as e:
                tqdm.tqdm.write(f"Error computing WER: {e}")
                wer_score = 1.0  # Assign worst score on error


            try:
                bleu_result = bleu_scorer.sentence_score(generated_text, [text_row])
                bleu_score = bleu_result.score / 100.0
            except Exception as e:
                tqdm.tqdm.write(f"Error computing BLEU: {e}")
                bleu_score = 0.0  # Assign worst score on error
            

            tqdm.tqdm.write(f"WER: {wer_score}")
            tqdm.tqdm.write(f"BLEU: {bleu_score}")
            tqdm.tqdm.write(f"Processed: {ref_speaker_id} + {target_speaker_id} | Speaker Similarity: {speaker_similarity_an_ref:.4f} | WER: {wer_score} | BLEU: {bleu_score}")

            

            local_metadata.append({
                "ref_file": str(ref_speaker_id),  # Convert NumPy int64 to Python int
                "target_file": str(target_speaker_id),  # Convert NumPy int64 to Python int
                "output_file": out_name,
                "output_path": key,
                "hdf5_key": key,
                "gender":target_gender,
                "WER": float(wer_score),  # Convert to Python float
                "BLEU": float(bleu_score),  # Convert to Python float
                "speaker_similarity_an_ref": float(speaker_similarity_an_ref),
                "speaker_similarity_an_target": float(speaker_similarity_an_target),
                "language": str(lan),  # Convert to Python str
                "orig_target_path": f"{str(target_store_ids)}",
                'generated_text':generated_text.lower(),
                'transcript_text':predicted_text.lower(),
                'target_text':text_row.lower(),
                'predicted_gender':target_gender,
                "sim_target_to_ref": target_to_ref,
                "ref_store_ids":ref_store_ids
            })
            
            pair_id = f"{str(ref_speaker_id)}_{str(target_speaker_id)}_{store_id}"
            
            while True:
                try:
                    checkpoint = load_checkpoint(checkpoint_path)  
                    checkpoint["completed"].append(pair_id)
                    checkpoint["metadata"].append(local_metadata[-1])  
                    save_checkpoint_with_file_lock(checkpoint_path, checkpoint["completed"], checkpoint["metadata"])
                    break  
                except Exception as e:
                    tqdm.tqdm.write(f"Error loading checkpoint: {e}")
                    time.sleep(30)  
    return local_metadata

def main(model_name: str, output_dir: str, name:str , ref_hdf5_path: str  , target_hdf5_path: str, ref_df = None, target_df = None,csv_output_dir: str = './csv_output/' ):

    os.makedirs(csv_output_dir, exist_ok=True)

    refernce_speaker_ids = ref_df.speaker_id.unique()
    checkpoint_filename = f"checkpoint_{model_name}_{name}.json"
    checkpoint_path = os.path.join(csv_output_dir, checkpoint_filename)

    checkpoint = load_checkpoint(checkpoint_path)
    #metadata = checkpoint["metadata"]  # Start with existing metadata

    output_dir = os.path.join(output_dir, f"{model_name}_{name}")
    os.makedirs(output_dir, exist_ok=True)
    paths = ModelPaths()
    tts, model, train_model, config = load_tts_and_trainer(paths)

    with ThreadPoolExecutor(max_workers=2) as executor:  
        futures = [
            executor.submit(process_reference, ref_speaker_id, model,  output_dir, train_model, config, model_name, tts, checkpoint_path,ref_df, target_df,ref_hdf5_path, target_hdf5_path)
            for ref_speaker_id in refernce_speaker_ids
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="References"):
            future.result()


    # Save metadata to a CSV
    final_checkpoint = load_checkpoint(checkpoint_path)
    df = pd.DataFrame(final_checkpoint["metadata"])
    df.to_csv(os.path.join(csv_output_dir, f"metadata_{model_name}_{name}.csv"), index=False)
    tqdm.tqdm.write(f"Metadata saved to metadata_{model_name}_{name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TTS anonymization')
    parser.add_argument('--languages', nargs='+', default=['de', 'en'], 
                        help='Languages to process (default: de en)')
    
    ref_path = os.getenv('ref_path')
    target_path = os.getenv('target_path')
    ref_hdf5_path = os.getenv('ref_hdf5_path')
    target_hdf5_path = os.getenv('target_hdf5_path')
    args = parser.parse_args()

    ref_df = pd.read_csv(ref_path)
    target_df = pd.read_csv(target_path)

    ref_df = ref_df[ref_df['lang'].isin(args.languages)]

    
    try:
        target_df = target_df[target_df['lang'].isin(args.languages)]
    except:
        target_df = target_df  
    
    #target_df = target_df.groupby('speaker_id', group_keys=False).apply(lambda x: x.sample(n=min(len(x), max(1, 100 // target_df['speaker_id'].nunique())), random_state=42)).head(100)

    output_dir = os.getenv('OUTPUT_PATH') if 'OUTPUT_PATH' in os.environ else '/cluster/home/muletrom/result/'
    main("xtts2_forward_iteration", 
        output_dir, 
        name=f'final_run_{target_path.split("/")[-1].replace(".csv", "")}_{ref_path.split("/")[-1].replace(".csv", "")}',
        ref_hdf5_path=ref_hdf5_path, 
        target_hdf5_path=target_hdf5_path,
        ref_df=ref_df, target_df=target_df,
        csv_output_dir=os.path.join(output_dir,'csv_outputs'))
   