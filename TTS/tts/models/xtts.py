import os
from dataclasses import dataclass

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit



from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager, LanguageManager
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.io import load_fsspec
from jiwer import wer
import re
from sacrebleu import BLEU
bleu_scorer = BLEU(effective_order=True)
import h5py
import numpy as np
import unicodedata

init_stream_support()


def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel

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

def load_audio(audiopath, orig_sr, hdf5_path=None):
    """Load audio and resample to target sample rate
    
    Args:
        audiopath: Path or HDF5 key
        target_sr: Target sample rate to resample to
        hdf5_path: If provided, load from HDF5
    """
    target_sr = 22050
    if hdf5_path is not None:
        audio = load_audio_hdf5(hdf5_path, str(audiopath))
    else:
        audio, orig_sr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)


    if orig_sr != target_sr:
        audio = torchaudio.functional.resample(audio, orig_sr, target_sr)

    # Validation
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    audio = audio.clip(-1, 1)
    
    return audio


def load_audio_hdf5(hdf5_path, file_key):
        with h5py.File(hdf5_path, "r") as hdf5_file:
            audio = hdf5_file[file_key][:]
        audio_tensor = torch.FloatTensor(audio)
        return audio_tensor

def pad_or_truncate(t, length):
    """
    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.

    Args:
        t (torch.Tensor): The input tensor to be padded or truncated.
        length (int): The desired length of the tensor.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000


@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


class Xtts(BaseTTS):
    """ⓍTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint  # TODO: check if this is even needed
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size

        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            decoder_input_dim=self.args.decoder_input_dim,
            d_vector_dim=self.args.d_vector_dim,
            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]

                # if the chunk is too short ignore it 
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
        hdf5_path=None
    ):
        """Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 24000.
        """
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        embedding_scores = []
        
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr, hdf5_path=hdf5_path)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            """speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)"""

            audios.append(audio)

            # get energy
            energy = torch.mean(audio ** 2).item()
            embedding_scores.append(energy)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        """if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)"""
        """weights = torch.tensor(embedding_scores) / sum(embedding_scores)
        speaker_embeddings_stacked = torch.stack(speaker_embeddings)
        weights_reshaped = weights.to(speaker_embeddings_stacked.device).view(-1, 1, 1, 1)

        speaker_embedding = (speaker_embeddings_stacked * weights_reshaped).sum(dim=0)"""

        speaker_embedding = self.get_speaker_embedding(full_audio, load_sr)

        return gpt_cond_latents, speaker_embedding

    def synthesize(self, text, config, speaker_wav, language, speaker_id=None, **kwargs):
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (XttsConfig): Config with inference parameters.
            speaker_wav (list): List of paths to the speaker audio files to be used for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
            return self.inference(text, language, gpt_cond_latent, speaker_embedding, **settings)
        settings.update({
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_len": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        })
        return self.full_inference(text, speaker_wav, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Cloning
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3
                seconds long.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used
                else the first `gpt_cond_len` secs is used. Defaults to 30 seconds.

            gpt_cond_chunk_len: (int) Chunk length used for cloning. It must be <= `gpt_cond_len`.
                If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to 6 seconds.

            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
        )

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        """Handle chunk formatting in streaming mode"""
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
        if wav_overlap is not None:
            # cross fade the overlap section
            if overlap_len > len(wav_chunk):
                # wav_chunk is smaller than overlap_len, pass on last wav_gen
                if wav_gen_prev is not None:
                    wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) :]
                else:
                    # not expecting will hit here as problem happens on last chunk
                    wav_chunk = wav_gen[-overlap_len:]
                return wav_chunk, wav_gen, None
            else:
                crossfade_wav = wav_chunk[:overlap_len]
                crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
                wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
                wav_chunk[:overlap_len] += crossfade_wav

        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return wav_chunk, wav_gen_prev, wav_overlap

    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # Streaming
        stream_chunk_size=20,
        overlap_wav_len=1024,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            fake_inputs = self.gpt.compute_embeddings(
                gpt_cond_latent.to(self.device),
                text_tokens,
            )
            gpt_generator = self.gpt.get_generator(
                fake_inputs=fake_inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                length_penalty=float(length_penalty),
                repetition_penalty=float(repetition_penalty),
                output_attentions=False,
                output_hidden_states=True,
                **hf_generate_kwargs,
            )

            last_tokens = []
            all_latents = []
            wav_gen_prev = None
            wav_overlap = None
            is_end = False

            while not is_end:
                try:
                    x, latent = next(gpt_generator)
                    last_tokens += [x]
                    all_latents += [latent]
                except StopIteration:
                    is_end = True

                if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                    gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                    if length_scale != 1.0:
                        gpt_latents = F.interpolate(
                            gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                        ).transpose(1, 2)
                    wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                    wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                        wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len
                    )
                    last_tokens = []
                    yield wav_chunk

    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):  # pylint: disable=unused-argument
        return Xtts(config)

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))["model"]
        # remove xtts gpt trainer extra keys
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=True,
        use_deepspeed=False,
        speaker_file_path=None,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.

        Args:
            config (dict): The configuration dictionary for the model.
            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.
            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.
            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.

        Returns:
            None
        """

        model_path = checkpoint_path or os.path.join(checkpoint_dir, "model.pth")
        vocab_path = vocab_path or os.path.join(checkpoint_dir, "vocab.json")

        if speaker_file_path is None and checkpoint_dir is not None:
            speaker_file_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

        self.language_manager = LanguageManager(config)
        self.speaker_manager = None
        if speaker_file_path is not None and os.path.exists(speaker_file_path):
            self.speaker_manager = SpeakerManager(speaker_file_path)

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        # deal with v1 and v1.1. V1 has the init_gpt_for_inference keys, v1.1 do not
        try:
            self.load_state_dict(checkpoint, strict=strict)
        except:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache)
            self.load_state_dict(checkpoint, strict=strict)

        if eval:
            self.hifigan_decoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
            self.gpt.eval()

    def train_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )


    @torch.no_grad()
    def forward_from_audios_and_text(
        self,
        lang,
        text,
        target_sample,
        ref_sample,
        train_model,
        max_conditioning_length,
        min_conditioning_length,
    ):
        device = self.device 

        o = self.prep_batch(
            lang,
            text,
            target_sample,
            ref_sample,
            train_model,
            max_conditioning_length,
            min_conditioning_length
        )

        wav_audio_codes = self.forward(o, ref_sample, train_model)

        return wav_audio_codes


    def forward(self, o, ref_sample, train_model, ref_hdf5_path=None):
        device = self.device

        cond_mels = o["cond_mels"].to(device)
        text_inputs = o["text_inputs"].to(device)
        text_lengths = o["text_lengths"].to(device)
        audio_codes = o["audio_codes"].to(device)
        wav_lengths = o["wav_lengths"].to(device)
        cond_lens = o["cond_lens"].to(device)
        train_model = train_model.to(device)

        train_model.training = False
        with torch.no_grad():
            (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
                audio_path=ref_sample,
                gpt_cond_len=45,
                gpt_cond_chunk_len=15,
                max_ref_length=45,
                sound_norm_refs=False,
                hdf5_path=ref_hdf5_path if ref_hdf5_path is not None else None
            )

            gpt_latents = self.gpt(
                text_inputs,
                text_lengths,
                audio_codes,
                wav_lengths,
                cond_latents=gpt_cond_latent,
                return_attentions=False,
                return_latent=True,
            )

            gpt_latents.shape, speaker_embedding.shape
                

        wav_audio_codes = self.hifigan_decoder(gpt_latents.clone(), g=speaker_embedding.clone()).cpu().squeeze()

        return {
            "wav": wav_audio_codes.cpu().squeeze().numpy(),
        }
    
    @torch.no_grad()
    def forward_from_batch(self, o, ref_samples, train_model, ref_hdf5_path=None):
        device = self.device

        text_inputs = o["text_inputs"].to(device)
        text_lengths = o["text_lengths"].to(device)
        audio_codes = o["audio_codes"].to(device)
        wav_lengths = o["wav_lengths"].to(device)
        cond_lens = o["cond_lens"].to(device)
        train_model = train_model.to(device)

        train_model.training = False
        with torch.no_grad():
            batch_size = text_inputs.shape[0]
            
            # Process conditioning latents for each batch item
            gpt_cond_latents_list = []
            speaker_embeddings_list = []
            gpt_latents_list = []
            
            # Handle both single ref_sample and list of ref_samples
            if isinstance(ref_samples, list):
                ref_list = ref_samples
            else:
                ref_list = [ref_samples] * batch_size
            
            # Process each sample INDEPENDENTLY through GPT to match single-mode behavior
            for idx in range(batch_size):
                ref_sample = ref_list[idx]
                
                # Get conditioning for this sample
                (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
                    audio_path=ref_sample,
                    gpt_cond_len=45,
                    gpt_cond_chunk_len=15,
                    max_ref_length=45,
                    sound_norm_refs=False,
                    hdf5_path=ref_hdf5_path if ref_hdf5_path is not None else None
                )
                gpt_cond_latents_list.append(gpt_cond_latent)
                speaker_embeddings_list.append(speaker_embedding)
                
                # Process this sample ALONE through GPT (not batched)
                # Extract single sample from batch
                text_input_single = text_inputs[idx:idx+1]  # Keep batch dim
                text_length_single = text_lengths[idx:idx+1]
                audio_code_single = audio_codes[idx:idx+1]
                wav_length_single = wav_lengths[idx:idx+1]
                
                # Forward through GPT with single sample
                gpt_latent_single = self.gpt(
                    text_input_single,
                    text_length_single,
                    audio_code_single,
                    wav_length_single,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )
                gpt_latents_list.append(gpt_latent_single)
            
            max_latent_len = max(latent.shape[1] for latent in gpt_latents_list)
            gpt_latents_padded = []
            for latent in gpt_latents_list:
                if latent.shape[1] < max_latent_len:
                    # Pad sequence dimension (dim 1)
                    pad_amount = max_latent_len - latent.shape[1]
                    latent_padded = F.pad(latent, (0, 0, 0, pad_amount))  # Pad end of dim 1
                    gpt_latents_padded.append(latent_padded)
                else:
                    gpt_latents_padded.append(latent)
            
            # Stack latents back into batch
            gpt_latents = torch.cat(gpt_latents_padded, dim=0)   # [B, T, 1024]
            
            # Stack speaker embeddings
            speaker_embeddings_squeezed = [se.squeeze(0) for se in speaker_embeddings_list]
            speaker_embedding = torch.stack(speaker_embeddings_squeezed)

        wav_audio_codes = self.hifigan_decoder(gpt_latents.clone(), g=speaker_embedding.clone()).cpu()
        wav_audio_codes = wav_audio_codes.squeeze(1)  # [B, output_length]

        # wav_lengths contains the ORIGINAL audio code lengths (before padding to batch max)
        # Convert to output audio lengths using code_stride_len
        code_stride_len = self.args.gpt_code_stride_len
        actual_output_lengths = (wav_lengths.cpu() * code_stride_len).tolist()

        return {
            "wav": wav_audio_codes.numpy(),
            "wav_lengths": actual_output_lengths,
        }
    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["text_lengths"] = torch.stack(batch["text_lengths"])
        batch["conditioning"] = torch.stack(batch["conditioning"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])
        batch["cond_idxs"] = torch.stack(batch["cond_idxs"])

        if torch.any(batch["cond_idxs"].isnan()):
            batch["cond_idxs"] = None

        if torch.any(batch["cond_lens"].isnan()):
            batch["cond_lens"] = None

        max_text_len = batch["text_lengths"].max()
        max_wav_len = batch["wav_lengths"].max()

        # create padding tensors
        text_padded = torch.IntTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        text_padded = text_padded.zero_()
        wav_padded = wav_padded.zero_()
        for i in range(B):
            text = batch["text"][i]
            text_padded[i, : batch["text_lengths"][i]] = torch.IntTensor(text)
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded
        batch["padded_text"] = text_padded
        return batch
    
    
    def prep_batch(self, lang,
        text,
        target_sample,
        ref_sample,
        train_model,
        max_conditioning_length,
        min_conditioning_length,
        target_hdf5_path=None,
        ref_hdf5_path=None,
        target_sample_rate=22050
        ):
        from TTS.tts.layers.xtts.trainer.dataset import get_prompt_slice
        device = self.device 
        if target_hdf5_path is not None:
            wav = load_audio(target_sample, target_sample_rate,hdf5_path=target_hdf5_path)
        else:
            wav = load_audio(target_sample, target_sample_rate)
        
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        tokens = self.tokenizer.encode(text, lang)#train_model.xtts.tokenizer.encode(text, lang)
        tseq = torch.IntTensor(tokens)

        if isinstance(ref_sample, list):
            conds, cond_lens = [], []
            for ref in ref_sample:
                cond, cond_len, _ = get_prompt_slice(
                    ref, max_conditioning_length, min_conditioning_length, target_sample_rate, True, hdf5_path=ref_hdf5_path if ref_hdf5_path is not None else None
                )
                conds.append(cond)
                cond_lens.append(cond_len)
            # Example: average conditioning
            cond = torch.stack(conds).mean(dim=0)
            cond_len = int(sum(cond_lens) / len(cond_lens))
        else:
            cond, cond_len, _ = get_prompt_slice(
                ref_sample, max_conditioning_length, min_conditioning_length, target_sample_rate, True, hdf5_path=ref_hdf5_path if ref_hdf5_path is not None else None
            )

        cond_idxs = torch.nan

        sample = {
            # 'real_text': text,
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": target_sample,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long)
            if cond_len is not torch.nan
            else torch.tensor([cond_len]),
            "cond_idxs": torch.tensor(cond_idxs) if cond_idxs is not torch.nan else torch.tensor([cond_idxs]),
        }

        collated = self.collate_fn([sample])
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                collated[k] = v.to(device)
        o = train_model.format_batch_on_device(collated)
        return o

    def prep_batch_multiple(self, langs, 
                    texts,
                    target_samples,
                    ref_samples,
                    train_model,
                    max_conditioning_length,
                    min_conditioning_length,
                    target_hdf5_path=None,
                    ref_hdf5_path=None,
                    target_sample_rate=22050):
        from TTS.tts.layers.xtts.trainer.dataset import get_prompt_slice
        
        batch_samples = []
        device = self.device 
        max_conditioning_length = int(max_conditioning_length)
        min_conditioning_length = int(min_conditioning_length)

        # Process each sample and collect into batch_samples list
        for text, target_sample, ref_sample, lang in zip(texts, target_samples, ref_samples, langs):
            
            # Load audio
            wav = load_audio(target_sample, target_sample_rate, hdf5_path=target_hdf5_path)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            # Tokenize text
            tokens = self.tokenizer.encode(text, lang)
            tseq = torch.IntTensor(tokens)

            # Get conditioning
            if isinstance(ref_sample, list):
                conds, cond_lens = [], []
                for ref in ref_sample:
                    cond, cond_len, _ = get_prompt_slice(
                        ref, max_conditioning_length, min_conditioning_length, 22050, True, 
                        hdf5_path=ref_hdf5_path
                    )
                    conds.append(cond)
                    cond_lens.append(cond_len)
                cond = torch.stack(conds).mean(dim=0)
                cond_len = int(sum(cond_lens) / len(cond_lens))
            else:
                cond, cond_len, _ = get_prompt_slice(
                    ref_sample, max_conditioning_length, min_conditioning_length, 22050, True, 
                    hdf5_path=ref_hdf5_path
                )

            # Create sample dict
            sample = {
                "text": tseq,
                "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
                "wav": wav,
                "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
                "filenames": target_sample,
                "conditioning": cond.unsqueeze(1),
                "cond_lens": torch.tensor(cond_len, dtype=torch.long) if cond_len is not torch.nan else torch.tensor([cond_len]),
                "cond_idxs": torch.tensor(torch.nan),
            }
            batch_samples.append(sample)
        
        # Let collate_fn handle all the batching
        collated = self.collate_fn(batch_samples)
        
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                collated[k] = v.to(device)
        
        # Format for the model
        o = train_model.format_batch_on_device(collated)
        return o


    @torch.no_grad()
    def forward_iteration(self, lang,
            text,
            target_sample,
            ref_sample,
            train_model,
            max_conditioning_length,
            min_conditioning_length,
            tts,
            ecapa,
            asr_model,
            n = 5):

        # for score calculation pre calculate target and ref embeddings
        target_audio = self.resample_audio_16k(target_sample)
        ref_audio = self.resample_audio_16k(ref_sample if isinstance(ref_sample, str) else ref_sample[0])
        ecapa_device = next(ecapa.parameters()).device
        ref_emb = ecapa(ref_audio.to(device=ecapa_device))
        tar_emb = ecapa(target_audio.to(device=ecapa_device))

        o = self.prep_batch(
            lang,
            text,
            target_sample,
            ref_sample,
            train_model.to(self.device),
            max_conditioning_length,
            min_conditioning_length
        )
        

        save_wavs =  {}
        save_scores = {"bleu_scores": [], "wer_scores": [],"target_cosine_similarities": [],"reference_cosine_similarities": []}
        audio_codes = []
        quality_scores = []
        audios = []

        for i in range(0, n):
            
            wav = self.forward(o, ref_sample, train_model)

            # save wav path for each iteration
            tts.synthesizer.save_wav(wav=wav['wav'], path=f"/home/romolo/VT1/coqui-tts/data/outputs/models/iterate_output_{i}.wav")

            # save wav path for each iteration          
            save_wavs[f"wav_{i}"] = f"/home/romolo/VT1/coqui-tts/data/outputs/models/iterate_output_{i}.wav"

            # compute scores
            wav_for_asr = librosa.resample(wav['wav'], orig_sr=24000, target_sr=16000)
            #wav_for_asr_float = wav_for_asr.astype('float32')
            
            for i in range(3):
                try:
                    generated_text = asr_model.transcribe(anonymized_wav_res,language=lan)['text']
                    if generated_text != None:
                        break
                except Exception as e:
                    tqdm.tqdm.write(f"Error in ASR transcription: {e}")
                    generated_text = ""
                    time.sleep(1)  # wait before retrying
            
    
            try:
                wer_score = wer(text, generated_text)
                break
            except Exception as e:
                tqdm.tqdm.write(f"Error computing WER: {e}")
                wer_score = 1.0  # Assign worst score on error
                time.sleep(1)  # wait before retrying
            

            try:
                bleu_score = bleu_scorer.sentence_score(generated_text,[text])["score"]/100.0
                break
            except Exception as e:
                tqdm.tqdm.write(f"Error computing BLEU: {e}")
                bleu_score = 0.0  # Assign worst score on error
  # wait before retrying



            save_scores["bleu_scores"].append(bleu_score)
            save_scores["wer_scores"].append(wer_score)

            # compute cosine similarities
            synth_audio = self.resample_audio_16k(f"/home/romolo/VT1/coqui-tts/data/outputs/models/iterate_output_{i}.wav")

            synth_emb = ecapa(synth_audio.to(device=self.device))
            save_scores["target_cosine_similarities"].append(torch.cosine_similarity(tar_emb, synth_emb))
            save_scores["reference_cosine_similarities"].append(torch.cosine_similarity(ref_emb, synth_emb))
            print(f"Iteration {i}: BLEU: {bleu_score}, WER: {wer_score}, Target Cosine: {save_scores['target_cosine_similarities'][-1]}, Reference Cosine: {save_scores['reference_cosine_similarities'][-1]}")
            quality_score = (
                0.05 * (1 - wer_score) +          # Lower WER is better
                0.05 * bleu_score +               # Higher BLEU is better
                0.30 * (1 - save_scores["target_cosine_similarities"][-1]) +         # Lower target similarity is better (voice conversion)
                0.60 * save_scores["reference_cosine_similarities"][-1]                    # Higher ref similarity is better
            )
            quality_scores.append(quality_score.item())
            audios.append(wav['wav'])

            # compute new audio codes for next iteration
            if i < n -1:
                o = self.prep_batch(
                lang,
                text,
                f"/home/romolo/VT1/coqui-tts/data/outputs/models/iterate_output_{i}.wav",
                ref_sample,
                train_model.to(self.device),
                max_conditioning_length,
                min_conditioning_length
                )

                audio_codes.append(o["audio_codes"])

        return save_wavs, save_scores, quality_scores, audios
   

    @torch.no_grad()
    def forward_iteration_hdf5(self, lang,
            text,
            target_sample,
            ref_sample,
            train_model,
            max_conditioning_length,
            min_conditioning_length,
            tts,
            ecapa,
            asr_model,
            target_hdf5_path,
            ref_hdf5_path,
            n = 5,
            ):
        
        # for score calculation pre calculate target and ref embeddings
        target_audio = self.resample_audio_16k(target_sample, hdf5_path=target_hdf5_path)
        ref_audio = self.resample_audio_16k(ref_sample if isinstance(ref_sample, str) else ref_sample[0], hdf5_path=ref_hdf5_path)
        ecapa_device = next(ecapa.parameters()).device
        ref_emb = ecapa(ref_audio.to(device=ecapa_device))
        tar_emb = ecapa(target_audio.to(device=ecapa_device))

        o = self.prep_batch(
            lang,
            text,
            target_sample,
            ref_sample,
            train_model.to(self.device),
            max_conditioning_length,
            min_conditioning_length,
            target_hdf5_path=target_hdf5_path,
            ref_hdf5_path=ref_hdf5_path
        )

        save_wavs = {}
        save_scores = {"bleu_scores": [], "wer_scores": [], "target_cosine_similarities": [], "reference_cosine_similarities": []}
        audio_codes = []
        quality_scores = []
        audios = []
        
        import tempfile
        import time
        
        PROJECT_TEMP_DIR = os.getenv("PROJECT_TEMP_DIR", "./temp/")
        os.makedirs(PROJECT_TEMP_DIR, exist_ok=True)
        temp_hdf5_fd, temp_hdf5_path = tempfile.mkstemp(suffix='.hdf5', prefix='iter_temp_', dir=PROJECT_TEMP_DIR)
        os.close(temp_hdf5_fd) 

        try:
            
            for i in range(0, n):

                
                wav = self.forward(o, ref_sample, train_model, ref_hdf5_path=ref_hdf5_path)

                # Ensure file handle is properly closed by using context manager
                for i in range(3):
                    try:
                        with h5py.File(temp_hdf5_path, 'a') as hdf5_temp:
                            iter_key = f'iteration_{i}'
                            if iter_key in hdf5_temp:
                                del hdf5_temp[iter_key]
                            hdf5_temp.create_dataset(iter_key, data=wav['wav'])
                            hdf5_temp[iter_key].attrs['sr'] = 24000
                            break
                    except Exception as e:
                        print(f"Error writing to temp HDF5 file {temp_hdf5_path}: {e}")
                        time.sleep(1)
                        raise e
                # File is closed here
                    
                wav_for_asr = librosa.resample(wav['wav'], orig_sr=24000, target_sr=16000)
                #wav_for_asr_float = wav_for_asr.astype('float32')
                generated_text = asr_model.transcribe(wav_for_asr, language=lang)["text"].lower()
                generated_text = clean_text(generated_text).lower()
                text = clean_text(text).lower()
                try:
                    bleu_score = bleu_scorer.sentence_score(generated_text,[text]).score/100.0
                except:
                    bleu_score = 0.0
                try:
                    wer_score = wer(reference=text, hypothesis=generated_text)
                except:
                    wer_score = 1.0
                
                save_scores["bleu_scores"].append(bleu_score)
                save_scores["wer_scores"].append(wer_score)

                # compute cosine similarities
                synth_emb = ecapa(torch.tensor(wav_for_asr).unsqueeze(0).to(device=self.device))
                save_scores["target_cosine_similarities"].append(torch.cosine_similarity(tar_emb, synth_emb))
                save_scores["reference_cosine_similarities"].append(torch.cosine_similarity(ref_emb, synth_emb))

                print(f"Iteration {i}: BLEU: {bleu_score}, WER: {wer_score}, Target Cosine: {save_scores['target_cosine_similarities'][-1]}, Reference Cosine: {save_scores['reference_cosine_similarities'][-1]}")
                try:
                    if (1 - wer_score) +(1 - save_scores["target_cosine_similarities"][-1]) == 0:
                        quality_score = 0.0
                    else:
                        quality_score = (
                            2*(
                                (1 - wer_score) *(1 - save_scores["target_cosine_similarities"][-1])
                            )/(
                                (1 - wer_score) +(1 - save_scores["target_cosine_similarities"][-1])
                            )
                        )
                except:
                    quality_score = 0.0
                quality_scores.append(float(quality_score))
                audios.append(wav['wav'])

                # compute new audio codes for next iteration
                if i < n - 1:
                    o = self.prep_batch(
                        lang,
                        text,
                        f'iteration_{i}',
                        ref_sample,
                        train_model.to(self.device),
                        max_conditioning_length,
                        min_conditioning_length,
                        target_hdf5_path=temp_hdf5_path,
                        ref_hdf5_path=ref_hdf5_path,
                        target_sample_rate=24000
                    )

                        
                    
        finally:
            # Robust cleanup
            try:
                if os.path.exists(temp_hdf5_path):
                    os.unlink(temp_hdf5_path)
                    print(f"Cleaned up temp HDF5 file: {temp_hdf5_path}")
            except OSError as e:
                print(f"Warning: Could not delete temp HDF5 file {temp_hdf5_path}: {e}")
                
        return save_wavs, save_scores, quality_scores, audios


    @torch.no_grad()
    def forward_iteration_hdf5_with_batch(self, langs,
            texts,
            target_samples,
            ref_samples,
            train_model,
            max_conditioning_length,
            min_conditioning_length,
            tts,
            ecapa,
            asr_model,
            target_hdf5_path,
            ref_hdf5_path,
            n = 5,
            predicted_texts = None
            ):
        
        if predicted_texts == None:
            predicted_texts = texts

        ecapa_device = next(ecapa.parameters()).device
        batch_size = len(predicted_texts) 
        # for score calculation pre calculate target and ref embeddings
        target_audios = []

        for target_sample in target_samples:
            target_audio = self.resample_audio_16k(target_sample, hdf5_path=target_hdf5_path)
            target_audios.append(target_audio)

        ref_audios = []
        for ref_sample in ref_samples:
            ref_audio = self.resample_audio_16k(ref_sample if isinstance(ref_sample, str) else ref_sample[0], hdf5_path=ref_hdf5_path)
            ref_audios.append(ref_audio)
        
        tar_embs = []
        for target_audio in target_audios:
            tar_emb = ecapa(target_audio.to(device=ecapa_device))
            tar_embs.append(tar_emb)
        tar_embs = torch.stack(tar_embs)  # Shape: [B, embedding_dim]
        
        ref_embs = []
        for ref_audio in ref_audios:
            ref_emb = ecapa(ref_audio.to(device=ecapa_device))
            ref_embs.append(ref_emb)
        ref_embs = torch.stack(ref_embs)

        o = self.prep_batch_multiple(
            langs,
            predicted_texts,
            target_samples,
            ref_samples,
            train_model.to(self.device),
            max_conditioning_length,
            min_conditioning_length,
            target_hdf5_path=target_hdf5_path,
            ref_hdf5_path=ref_hdf5_path
        )


        save_wavs = {}
        save_scores = {
            "bleu_scores": [[] for _ in range(batch_size)],
            "wer_scores": [[] for _ in range(batch_size)],
            "target_cosine_similarities": [[] for _ in range(batch_size)],
            "reference_cosine_similarities": [[] for _ in range(batch_size)]
        }

        audio_codes = [[] for _ in range(batch_size)]
        quality_scores = [[] for _ in range(batch_size)]
        audios = [[] for _ in range(batch_size)] 
        
        import tempfile
        import time
        
        PROJECT_TEMP_DIR = os.getenv("PROJECT_TEMP_DIR", "./temp/")
        os.makedirs(PROJECT_TEMP_DIR, exist_ok=True)
        temp_hdf5_fd, temp_hdf5_path = tempfile.mkstemp(suffix='.hdf5', prefix='iter_temp_', dir=PROJECT_TEMP_DIR)
        os.close(temp_hdf5_fd) 

        try:
            
            for i in range(0, n):
                wav = self.forward_from_batch(o, ref_samples, train_model, ref_hdf5_path=ref_hdf5_path)
                

                batch_size = len(texts)
                wav_lengths = wav.get("wav_lengths", [wav['wav'].shape[1]] * batch_size)
                for batch_idx in range(batch_size):

                    
                    lang = langs[batch_idx]
                    text = texts[batch_idx]
                    
                    actual_length = int(wav_lengths[batch_idx])
                    audio_sample = wav['wav'][batch_idx, :actual_length]

                    

                    with h5py.File(temp_hdf5_path, 'a') as hdf5_temp:
                        iter_key = f'sample_{batch_idx}_iteration_{i}'
                        if iter_key in hdf5_temp:
                            del hdf5_temp[iter_key]
                        hdf5_temp.create_dataset(iter_key, data=audio_sample)
                        hdf5_temp[iter_key].attrs['sr'] = 24000
                    
                    audios[batch_idx].append(audio_sample)

                    # File is closed here
                    wav_for_asr = librosa.resample(audio_sample if isinstance(audio_sample, np.ndarray) else audio_sample.numpy(), 
                               orig_sr=24000, target_sr=16000)
                    wav_for_asr_float = wav_for_asr.astype('float32')
                    generated_text = asr_model.transcribe(wav_for_asr_float, language=lang)["text"].lower()
                    generated_text = clean_text(generated_text)
                    text = clean_text(text)
                    bleu_score = bleu_scorer.sentence_score(generated_text,[text])["score"]/100.0
                    
                    wer_score = wer(reference=text.lower(), hypothesis=generated_text.lower())


                    # Compute cosine similarities with batch indexing
                    synth_emb = ecapa(torch.from_numpy(wav_for_asr).float().unsqueeze(0).to(device=ecapa_device))
                    # Squeeze all singleton dimensions from synth_emb to get [embedding_dim]
                    synth_emb = synth_emb.squeeze()
                    
                    tar_emb_single = tar_embs[batch_idx].squeeze()  # [192]
                    ref_emb_single = ref_embs[batch_idx].squeeze()  # [192]

                    tar_cos = torch.cosine_similarity(tar_emb_single.unsqueeze(0), synth_emb.unsqueeze(0)).squeeze().item()
                    ref_cos = torch.cosine_similarity(ref_emb_single.unsqueeze(0), synth_emb.unsqueeze(0)).squeeze().item()
                    
                    # save scores
                    save_scores["bleu_scores"][batch_idx].append(bleu_score)
                    save_scores["wer_scores"][batch_idx].append(wer_score)
                    save_scores["target_cosine_similarities"][batch_idx].append(tar_cos)
                    save_scores["reference_cosine_similarities"][batch_idx].append(ref_cos)
                    
                    quality_score = 2 *(1-wer_score) * (1 - tar_cos) /  ((1 - wer_score) + (1 - tar_cos)) if ((1 - wer_score) + (1 - tar_cos)) != 0 else 0.0
                    # Alternative simpler quality score

                    quality_scores[batch_idx].append(float(quality_score))
                    print(f"Sample {batch_idx}, Iteration {i}: BLEU: {bleu_score}, WER: {wer_score}, Quality: {quality_scores[batch_idx][-1]}")

                # compute new audio codes for next iteration
                if i < n - 1:
                    next_target_samples = [f'sample_{batch_idx}_iteration_{i}' for batch_idx in range(batch_size)]
                    o = self.prep_batch_multiple(
                        langs,
                        predicted_texts,
                        next_target_samples,
                        ref_samples,
                        train_model.to(self.device),
                        max_conditioning_length,
                        min_conditioning_length,
                        target_hdf5_path=temp_hdf5_path,
                        ref_hdf5_path=ref_hdf5_path,
                        target_sample_rate=24000
                    )

                        
                    
        finally:
            # Robust cleanup
            try:
                if os.path.exists(temp_hdf5_path):
                    os.unlink(temp_hdf5_path)
                    print(f"Cleaned up temp HDF5 file: {temp_hdf5_path}")
            except OSError as e:
                print(f"Warning: Could not delete temp HDF5 file {temp_hdf5_path}: {e}")
                
        return save_wavs, save_scores, quality_scores, audios

    def resample_audio_16k(self, audiopath,new_freq=16000,hdf5_path=None, orig_freq=22050):
        """Resample the given audio to 16kHz.
        Args:
            audiopath (str): Path to the audio file.
        Returns:
            Resampled audio tensor at 16kHz.
        """
        # if h5py path is given load from hdf5 file
        if hdf5_path is not None:
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                ref_audio = torch.tensor(hdf5_file[audiopath][:])
                # Try to get sample rate from attributes, otherwise use orig_freq parameter
                if 'sr' in hdf5_file[audiopath].attrs:
                    orig_freq = hdf5_file[audiopath].attrs['sr']
            
            # Ensure it's 2D for resampling
            if ref_audio.dim() == 1:
                ref_audio = ref_audio.unsqueeze(0)
            ref_audio = torchaudio.functional.resample(ref_audio, orig_freq=orig_freq, new_freq=new_freq)
            return ref_audio
        else:
            ref_audio, sr = torchaudio.load(audiopath)
            # Ensure it's 2D for resampling
            if ref_audio.dim() == 1:
                ref_audio = ref_audio.unsqueeze(0)
            ref_audio = torchaudio.functional.resample(ref_audio, orig_freq=sr, new_freq=new_freq)
            return ref_audio
    
