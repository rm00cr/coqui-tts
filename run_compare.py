from huggingface_hub import hf_hub_download
import torchaudio
import torch
import tqdm
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
references_dir = "/home/romolo/VT1/coqui-tts/test_data/Dataset/references_new"
targets_dir = "/home/romolo/VT1/coqui-tts/test_data/Dataset/references_new"
output_dir = "/home/romolo/VT1/prog/streamlit_prog/outputs/voice_sim/"


device = "cuda" if torch.cuda.is_available() else "cpu"


model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
ecapa2 = torch.jit.load(model_file, map_location=device)

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


speaker_embedding_cache = {}
def get_speaker_embedding(path):
    path_name = path[0].split('/')[-2]
    if path_name not in speaker_embedding_cache:
        audio = resample_audio(path).to(device)
        with torch.no_grad():
                embedding = ecapa2(audio)
                speaker_embedding_cache[path_name] = embedding.cpu()
        del audio
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return speaker_embedding_cache[path_name]


def calc_speaker_similarity(ref_audio_path, target_audio_path):
    # Move audio to the same device as the model

    ref_embedding = get_speaker_embedding(ref_audio_path)
    target_embedding = get_speaker_embedding(target_audio_path)

    # Compute cosine similarity
    with torch.no_grad():
        similarity = torch.cosine_similarity(ref_embedding, target_embedding)
        return similarity.item()


def process_references(ref_file: str, target_dir: str, output_dir: str):
    metadata = []

    ref_path = os.path.join(references_dir, ref_file)
    ref_speakers = os.listdir(ref_path)
    ref_speakers = [os.path.join(ref_path, spk) for spk in ref_speakers]
    ref_lang = ref_file.split('_')[0]

    for target_speaker_folder in tqdm.tqdm(os.listdir(targets_dir), leave=False, desc=f"{ref_file} targets"):
        target_path = os.path.join(targets_dir, target_speaker_folder)
        target_speakers = os.listdir(target_path)
        target_speakers = [os.path.join(target_path, spk) for spk in target_speakers]

        lang = target_speaker_folder.split('_')[0]
        """if lang != ref_lang:
            continue  # skip if languages do not match
        """
        similarity = calc_speaker_similarity(ref_speakers, target_speakers)
        tqdm.tqdm.write(f"Speaker similarity for {ref_file} and {target_speaker_folder}: {similarity}")
        metadata.append({
            "language": lang,
            "reference_file": ref_file,
            "target_file": target_speaker_folder,
            "speaker_similarity": similarity,
        })

    return metadata


def main(csv_name: str, output_dir: str = output_dir, name:str =f'{time.time()}'):
    metadata = []
    output_dir = os.path.join(output_dir, f"{csv_name}")
    os.makedirs(output_dir, exist_ok=True)

    ref_files = os.listdir(references_dir)
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
        futures = [
            executor.submit(process_references, ref_file, targets_dir, output_dir)
            for ref_file in ref_files
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="References"):
            metadata.extend(future.result())

    # Save metadata to a CSV
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, f"{csv_name}.csv"), index=False)
    tqdm.tqdm.write(f"Metadata saved to {csv_name}.csv")

if __name__ == "__main__":
    main("cosine_sim_ref_target_large", output_dir, name=f'{time.time()}')