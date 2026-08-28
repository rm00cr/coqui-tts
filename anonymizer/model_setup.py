import torch
import os
from dataclasses import dataclass, field
from TTS.api import TTS
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

from .download import default_model_dir

#: Where the XTTS v2 checkpoint files (model.pth, dvae.pth, mel_stats.pth, vocab.json,
#: config.json) live, resolved once at import for callers that want a plain string.
#: Prefer `default_model_dir()`, which re-reads $XTTS_MODEL_DIR on every call.
DEFAULT_CHECKPOINTS_DIR = default_model_dir()


@dataclass
class ModelPaths:
    #: Resolved per instance, not at class-definition time, so $XTTS_MODEL_DIR set after
    #: import is still honoured.
    checkpoints_out_path: str = field(default_factory=default_model_dir)
    out_path: str = "./"

    @property
    def dvae_checkpoint(self):
        return os.path.join(self.checkpoints_out_path, "dvae.pth")

    @property
    def mel_norm_file(self):
        return os.path.join(self.checkpoints_out_path, "mel_stats.pth")

    @property
    def tokenizer_file(self):
        return os.path.join(self.checkpoints_out_path, "vocab.json")

    @property
    def xtts_checkpoint(self):
        return os.path.join(self.checkpoints_out_path, "model.pth")

    @property
    def config_path(self):
        return os.path.join(self.checkpoints_out_path, "config.json")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_gpt_config(paths: ModelPaths) -> GPTTrainerConfig:
    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=paths.mel_norm_file,
        dvae_checkpoint=paths.dvae_checkpoint,
        xtts_checkpoint=paths.xtts_checkpoint,
        tokenizer_file=paths.tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    return GPTTrainerConfig(
        output_path=paths.out_path,
        model_args=model_args,
        epochs=1000,
        run_description="GPT XTTS training",
        audio=audio_config,
        model_param_stats=False,
        batch_size=32,
        batch_group_size=48,
        eval_batch_size=32,
        num_loader_workers=2,
        eval_split_max_size=256,
        eval_split_size=0.02,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        wandb_entity='RM',
        print_eval=False,
        datasets=None,
        shuffle=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=6e-05,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        use_h5=True,
    )

def load_tts_and_trainer(paths: ModelPaths, device=None):
    if device is None:
        device = get_device()
    tts = TTS(
        model_path=paths.checkpoints_out_path,
        config_path=paths.config_path,
        progress_bar=True
    ).to(device)
    tts.synthesizer.output_sample_rate = 24000
    model = tts.synthesizer.tts_model
    config = build_gpt_config(paths)
    train_model = GPTTrainer.init_from_config(config)
    train_model = train_model.to(device) 
    return tts, model, train_model, config

# Usage example:
# paths = ModelPaths()
# tts, model, train_model, config = load_tts_and_trainer(paths)