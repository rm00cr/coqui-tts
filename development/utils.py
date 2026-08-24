import torch
import numpy as np
import jiwer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchaudio
from TTS.tts.models.xtts import load_audio
import soundfile as sf
import os
import tempfile
import uuid
import threading
import time
from tqdm import tqdm
import noisereduce as nr
import librosa

def split_audio(audio, segment_length_samples):
    segments = []
    for start in range(0, audio.shape[-1], segment_length_samples):
        end = min(start + segment_length_samples, audio.shape[-1])
        segments.append(audio[..., start:end])
    return segments

def process_segments(segments, scoring_fn, threshold):
    accepted = []
    to_improve = []
    for i, seg in enumerate(segments):
        score = scoring_fn(seg)
        if score > threshold:
            accepted.append(seg)
        else:
            to_improve.append((i, seg))
    return accepted, to_improve

def stitch_segments(segments):
    # Convert all segments to tensors if they are numpy arrays
    segments = [torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg for seg in segments]
    return torch.cat(segments, dim=-1)

def get_unique_temp_filename(suffix='.wav'):
    """Generate a unique temporary filename in the system temp dir.

    Returns an absolute path so callers do not litter their working directory.
    """
    timestamp = str(int(time.time() * 1000000))  # Microsecond timestamp
    thread_id = str(threading.get_ident())
    unique_id = str(uuid.uuid4())[:8]
    return os.path.join(tempfile.gettempdir(), f"temp_{timestamp}_{thread_id}_{unique_id}{suffix}")

def assess_segment_quality(segment_audio, segment_text, target_speaker_emb, ref_speaker_emb, asr_model, ecapa_model, device=None):
    """Assess quality of a single segment"""
    if device is None:
        device = next(ecapa_model.parameters()).device

    # Convert to numpy for ASR processing
    audio_np = segment_audio.detach().cpu().numpy().squeeze()
    smoothie = SmoothingFunction().method4
    
    # 1. Speech Recognition Quality (WER)
    try:
        # Save segment to temporary file for ASR with unique name
        temp_filename = get_unique_temp_filename('.wav')
        sf.write(temp_filename, audio_np, 24000)
        transcribed = asr_model.transcribe(temp_filename)['text']
        
        # Clean up temp file
        os.unlink(temp_filename)
        
        wer_score = jiwer.wer(reference=segment_text.lower(), hypothesis=transcribed.lower())
        blue_score = sentence_bleu([segment_text.lower().split()], transcribed.lower().split(), smoothing_function=smoothie)
    except Exception as e:
        tqdm.write(f"ASR processing failed: {e}")
        wer_score = 1.0  # Worst case if ASR fails
        blue_score = 0.0
    
    # 2. Speaker Similarity - RESAMPLE TO 16KHZ FOR ECAPA
    try:
        # Resample segment audio to 16kHz for ECAPA
        segment_16k = resample_audio_16k(segment_audio, orig_freq=24000, new_freq=16000)
        segment_emb = ecapa_model(segment_16k.to(device=device))
        
        target_sim = torch.cosine_similarity(target_speaker_emb, segment_emb).item()
        ref_sim = torch.cosine_similarity(ref_speaker_emb, segment_emb).item()
    except Exception as e:
        tqdm.write(f"Speaker similarity calculation failed: {e}")
        target_sim = 0.0
        ref_sim = 0.0
    
    # Combine scores (customize weights as needed)
    quality_score = (
        0.05 * (1 - wer_score) +          # Lower WER is better
        0.05 * blue_score +               # Higher BLEU is better
        0.30 * (1 - target_sim) +         # Lower target similarity is better (voice conversion)
        0.60 * ref_sim                    # Higher ref similarity is better
    )
    
    return {
        'overall_quality': quality_score,
        'wer': wer_score,
        'blue': blue_score,
        'target_sim': target_sim,
        'ref_sim': ref_sim
    }


def split_text_by_segments(text, num_segments):
    """Split text into segments (simple word-based splitting)"""
    words = text.split()
    words_per_segment = len(words) // num_segments
    
    segments = []
    for i in range(num_segments):
        start_idx = i * words_per_segment
        if i == num_segments - 1:  # Last segment gets remaining words
            end_idx = len(words)
        else:
            end_idx = (i + 1) * words_per_segment
        
        segment_text = ' '.join(words[start_idx:end_idx])
        segments.append(segment_text)
    
    return segments


def improved_crossfade_stitch(segments, fade_samples=1024):  # Longer fade
    """Improved crossfade with better blending"""
    if len(segments) == 0:
        return np.array([])
    
    if len(segments) == 1:
        return segments[0]
    
    output = segments[0].astype(np.float32)
    
    for seg in segments[1:]:
        seg = seg.astype(np.float32)
        
        # Use longer fade for smoother transitions
        actual_fade = min(fade_samples, len(output) // 4, len(seg) // 4)  # Max 25% of segment
        
        if actual_fade > 0:
            fade_out = output[-actual_fade:]
            fade_in = seg[:actual_fade]
            
            # Use cosine crossfade instead of linear (smoother)
            fade_out_curve = np.cos(np.linspace(0, np.pi/2, actual_fade))
            fade_in_curve = np.sin(np.linspace(0, np.pi/2, actual_fade))
            
            crossfaded = fade_out * fade_out_curve + fade_in * fade_in_curve
            output = np.concatenate([output[:-actual_fade], crossfaded, seg[actual_fade:]])
        else:
            output = np.concatenate([output, seg])
    
    return output

def combine_segments_with_overlap(segments, overlap_samples=1024):
    """Combine segments using overlap and crossfade technique"""
    if len(segments) <= 1:
        return segments[0] if segments else np.array([])
    
    result = segments[0].astype(np.float32)
    
    for i, current_seg in enumerate(segments[1:], 1):
        current_seg = current_seg.astype(np.float32)
        
        # Create overlap region
        if len(result) >= overlap_samples and len(current_seg) >= overlap_samples:
            # Get overlapping sections
            prev_tail = result[-overlap_samples:]
            curr_head = current_seg[:overlap_samples]
            
            # Apply crossfade using numpy (not torch)
            fade_curve = np.linspace(0, 1, overlap_samples)
            blended = prev_tail * (1 - fade_curve) + curr_head * fade_curve
            
            # Combine using numpy concatenate
            result = np.concatenate([
                result[:-overlap_samples], 
                blended, 
                current_seg[overlap_samples:]
            ])
        else:
            # Fallback to simple concatenation
            result = np.concatenate([result, current_seg])
    
    return result


def stitch_segments_with_crossfade(segments):
    """Stitch audio segments back together with crossfading"""
    if len(segments) == 1:
        return segments[0]
    
    # Convert to numpy for processing
    segments_np = [seg.detach().cpu().numpy().squeeze() for seg in segments]
    
    # Use crossfade stitching
    result = combine_segments_with_overlap(segments_np, overlap_samples=1024)
    
    return torch.tensor(result)

def generate_text_segments_with_whisper(asr_model, segments, sample_rate=24000, debug=False):
    """Generate text segments by transcribing each audio segment individually"""
    text_segments = []
    
    for i, segment_audio in enumerate(segments):
        try:
            # Convert segment to numpy
            audio_np = segment_audio.detach().cpu().numpy().squeeze()
            
            # Save segment to temporary file for ASR with unique name
            temp_filename = get_unique_temp_filename('.wav')
            sf.write(temp_filename, audio_np, sample_rate)
            transcribed = asr_model.transcribe(temp_filename)['text']
            
            # Clean up temp file
            os.unlink(temp_filename)
            
            text_segments.append(transcribed.strip())
            if debug:
                tqdm.write(f"Segment {i} text: '{transcribed.strip()}'")
            
        except Exception as e:
            if debug:
                tqdm.write(f"Failed to transcribe segment {i}: {e}")
            # Fallback: use a generic text or empty string
            text_segments.append(f"segment {i}")
    
    return text_segments

def regenerate_segment_multiple_attempts(model, text_seg, segment_file, ref_speaker, train_model, max_conditioning_length, min_conditioning_length, target_speaker_emb, ref_speaker_emb, asr_model, ecapa_model, lang='en', max_attempts=3, target_threshold=0.8,debug=False):
    """Regenerate a segment multiple times and choose the best one"""
    
    best_segment = None
    best_quality = -1
    best_scores = None
    current_segment_file = segment_file  # Track the current input file
    temp_files_created = []  # Track temp files for cleanup
    
    if debug:
        tqdm.write(f"    Attempting {max_attempts} progressive regenerations...")

    for attempt in range(max_attempts):
        try:
            # Regenerate segment using the current best as input
            segment_result = model.forward_from_audios_and_text(
                lang, text_seg, current_segment_file, ref_speaker, 
                train_model, max_conditioning_length, min_conditioning_length
            )
            
            candidate_segment = torch.tensor(segment_result['wav'])
            
            # Save this attempt with unique filename
            attempt_file = get_unique_temp_filename(f'_attempt_{attempt + 1}.wav')
            sf.write(attempt_file, candidate_segment.detach().cpu().numpy().squeeze(), 24000)
            temp_files_created.append(attempt_file)
            
            # Assess quality of this attempt
            quality_info = assess_segment_quality(
                candidate_segment, text_seg, target_speaker_emb, ref_speaker_emb, asr_model, ecapa_model
            )
            
            current_quality = quality_info['overall_quality']
            if debug:
                tqdm.write(f"      Attempt {attempt + 1}: Quality={current_quality:.3f}, WER={quality_info['wer']:.3f}, BLEU={quality_info['blue']:.3f}, RefSim={quality_info['ref_sim']:.3f}")

            # Keep if it's the best so far
            if current_quality > best_quality:
                best_segment = candidate_segment
                best_quality = current_quality
                best_scores = quality_info
                
                # Update the input file for next iteration to build upon this improvement
                current_segment_file = attempt_file
                if debug:
                    tqdm.write(f"        → New best! Using this as input for next attempt")

                # Early stopping if we reach target threshold
                if current_quality >= target_threshold:
                    if debug:
                        tqdm.write(f"      ✓ Target quality reached on attempt {attempt + 1}!")
                    break
            else:
                if debug:
                    tqdm.write(f"        → No improvement, keeping previous best as input")
                # Don't update current_segment_file, keep using the best one
                
        except Exception as e:
            if debug:
                tqdm.write(f"      Attempt {attempt + 1} failed: {e}")
            continue
    
    # Clean up temporary files
    for temp_file in temp_files_created:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    return best_segment, best_quality, best_scores

def iterative_segment_refinement(model, target_speaker, ref_speaker, max_conditioning_length, min_conditioning_length, train_model, asr_model, ecapa_model, lang='en', threshold=0.7, max_attempts=3,regenerate=False,debug=False, device=None):
    if device is None:
        device = next(ecapa_model.parameters()).device
    # Step 1: Load target audio to get text and for embeddings
    target_audio = load_audio(target_speaker, 24000)
    text = asr_model.transcribe(target_speaker)['text']
    
    # Generate full audio using file paths
    wav = model.forward_from_audios_and_text(
        lang, text, target_speaker, ref_speaker, 
        train_model, max_conditioning_length, min_conditioning_length
    )
    
    # Convert to tensor if needed
    if isinstance(wav["wav"], np.ndarray):
        audio_tensor = torch.tensor(wav["wav"])
    else:
        audio_tensor = wav["wav"]
    
    # Get speaker embeddings for comparison
    target_audio_16k = resample_audio_16k(torch.tensor(target_audio), orig_freq=24000, new_freq=16000)
    ref_audio = load_audio(ref_speaker if isinstance(ref_speaker, str) else ref_speaker[0], 24000)
    ref_audio_16k = resample_audio_16k(torch.tensor(ref_audio), orig_freq=24000, new_freq=16000)

    target_speaker_emb = ecapa_model(target_audio_16k.to(device))
    ref_speaker_emb = ecapa_model(ref_audio_16k.to(device))

    # Step 2: Split into 3-second segments
    segment_length = 2 * 24000
    segments = split_audio(audio_tensor, segment_length)
    if debug:
        tqdm.write("Generating text segments with Whisper...")
    text_segments = generate_text_segments_with_whisper(asr_model, segments, 24000)
    
    if debug:
        tqdm.write(f"Original audio length: {audio_tensor.shape[-1] / 24000:.2f} seconds")
        tqdm.write(f"Number of segments: {len(segments)}")
    
    # Step 3: Assess each segment
    refined_segments = []
    segments_improved = 0
    total_attempts = 0
    temp_files_created = []  # Track all temp files for cleanup
    
    for i, (audio_seg, text_seg) in enumerate(zip(segments, text_segments)):
        original_seg_length = audio_seg.shape[-1]
        
        # Save segment file for regeneration with unique name
        segment_file = get_unique_temp_filename(f'_segment_{i}.wav')
        sf.write(segment_file, audio_seg.detach().cpu().numpy().squeeze(), 24000)
        temp_files_created.append(segment_file)
        
        # Assess original segment quality
        quality_info = assess_segment_quality(
            audio_seg, text_seg, target_speaker_emb, ref_speaker_emb, asr_model, ecapa_model
        )
        if debug:
            tqdm.write(f"Segment {i}: Text='{text_seg[:50]}{'...' if len(text_seg) > 50 else ''}'")
            tqdm.write(f"  Original Quality={quality_info['overall_quality']:.3f}, WER={quality_info['wer']:.3f}, BLEU={quality_info['blue']:.3f}, RefSim={quality_info['ref_sim']:.3f}")
        
        if quality_info['overall_quality'] < threshold:  # Bad segment
            if debug:
                tqdm.write(f"  🔄 Refining segment {i} (quality below {threshold})")
            
            # Try multiple regenerations
            best_segment, best_quality, best_scores = regenerate_segment_multiple_attempts(
                model, text_seg, segment_file, ref_speaker, 
                train_model, max_conditioning_length, min_conditioning_length,
                target_speaker_emb, ref_speaker_emb, asr_model, ecapa_model,
                lang, max_attempts, target_threshold=0.8
            )
            
            total_attempts += max_attempts
            
            if best_segment is not None:
                # Trim/pad to original length
                if best_segment.shape[-1] > original_seg_length:
                    best_segment = best_segment[..., :original_seg_length]
                    if debug:
                        tqdm.write(f"    Trimmed to {original_seg_length/24000:.2f}s")
                elif best_segment.shape[-1] < original_seg_length:
                    padding = original_seg_length - best_segment.shape[-1]
                    best_segment = torch.nn.functional.pad(best_segment, (0, padding))
                    if debug:
                        tqdm.write(f"    Padded to {original_seg_length/24000:.2f}s")
                
                improvement = best_quality - quality_info['overall_quality']
                if improvement > 0:
                    if debug:
                        tqdm.write(f"    ✓ Improved by {improvement:.3f} (final quality: {best_quality:.3f})")
                    segments_improved += 1
                    refined_segments.append(best_segment)
                else:
                    if debug:
                        tqdm.write(f"    ⚠️ No improvement (keeping original)")
                    refined_segments.append(audio_seg)
            else:
                if debug:
                    tqdm.write(f"    ❌ All attempts failed (keeping original)")
                refined_segments.append(audio_seg)
                
        else:  # Good segment - keep as is
            if debug:
                tqdm.write(f"  ✓ Good quality (keeping original)")
            refined_segments.append(audio_seg)
    
    # Clean up all temp files
    for temp_file in temp_files_created:
        try:
            os.unlink(temp_file)
        except:
            pass
    
    # Step 4: Stitch back together
    if debug:
        tqdm.write(f"\n📊 Summary:")
        tqdm.write(f"  - Segments processed: {len(segments)}")
        tqdm.write(f"  - Segments improved: {segments_improved}")
        tqdm.write(f"  - Total regeneration attempts: {total_attempts}")
    
    final_audio = stitch_segments_with_crossfade(refined_segments)



    if regenerate:
        temp_filename = get_unique_temp_filename('.wav')
        sf.write(temp_filename, final_audio, 24000)
        final_audio = model.forward_from_audios_and_text(
        lang, text, temp_filename, ref_speaker, 
        train_model, max_conditioning_length, min_conditioning_length
        )
        final_audio = final_audio["wav"]
        os.unlink(temp_filename)

    
    return final_audio


def resample_audio_16k(audio_tensor, orig_freq=24000, new_freq=16000):
    """Resample the given audio tensor to 16kHz.
    Args:
        audio_tensor (torch.Tensor): Audio tensor to resample.
        orig_freq (int): Original sample rate.
        new_freq (int): Target sample rate (default 16kHz for ECAPA).
    Returns:
        Resampled audio tensor at target frequency.
    """
    if orig_freq == new_freq:
        return audio_tensor
    
    # Ensure audio is 2D for resampling
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    resampled = torchaudio.functional.resample(audio_tensor, orig_freq=orig_freq, new_freq=new_freq)
    return resampled



def denoise_audio_to_temp(input_path):
    """
    Denoise audio and save to a temporary file
    
    Args:
        input_path: Path to input audio file
    
    Returns:
        str: Path to the temporary denoised audio file
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(input_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    # --- Trim silence using librosa ---
    trimmed, _ = librosa.effects.trim(audio_np, top_db=50)
    audio_np = trimmed.astype(np.float32)

    # --- Apply light denoising ---
    reduced_noise = nr.reduce_noise(
        y=audio_np,
        sr=sample_rate,
        stationary=False,
        prop_decrease=0.7,
        n_fft=1024
    )

    # Normalize
    reduced_noise = reduced_noise / (np.max(np.abs(reduced_noise)) + 1e-6)

    # --- Create temporary file - FIXED ---
    temp_path = get_unique_temp_filename_for_denoising('_denoised.wav')
    
    # --- Save to temporary file ---
    sf.write(temp_path, reduced_noise, sample_rate)
    
    return temp_path

def get_unique_temp_filename_for_denoising(suffix='_denoised.wav'):
    """Generate a unique temporary filename for denoising"""
    timestamp = str(int(time.time() * 1000000))
    thread_id = str(threading.get_ident())
    unique_id = str(uuid.uuid4())[:8]
    
    temp_dir = tempfile.gettempdir()
    filename = f"temp_{timestamp}_{thread_id}_{unique_id}{suffix}"
    return os.path.join(temp_dir, filename)

# Remove the duplicate definition at the end of your file
# Keep only the original get_unique_temp_filename() function:
# Add this to your utils.py file:

from contextlib import contextmanager

@contextmanager
def denoised_temp_file(input_path):
    """Context manager that automatically cleans up temporary denoised file"""
    temp_path = denoise_audio_to_temp(input_path)
    try:
        yield temp_path
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass