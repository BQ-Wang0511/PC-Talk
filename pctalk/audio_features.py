"""In-process extraction of PC-Talk's audio-visual features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import torch
from scipy import signal

from .models.audio_visual_encoder import AudioVisualEncoder

from .config import AudioConfig


class AudioFeatureExtractor:
    """Extract the 512-D features used to train PC-Talk.

    It replaces the research preprocessing script and its temporary feature
    array with a reusable in-process component.
    """

    _N_FFT = 800
    _HOP_SIZE = 200
    _WIN_SIZE = 800
    _NUM_MELS = 80
    _FMIN = 55
    _FMAX = 7600
    _PREEMPHASIS = 0.97
    _MIN_LEVEL_DB = -100
    _REF_LEVEL_DB = 20
    _MAX_ABS_VALUE = 4.0

    def __init__(self, config: AudioConfig, device: torch.device):
        self.config = config
        self.device = device
        checkpoint = Path(config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Audio encoder checkpoint not found: {checkpoint}")

        self.model = AudioVisualEncoder().to(device)
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(str(checkpoint), map_location="cpu")
        state = self._normalise_state_dict(state)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)

        self._mel_basis = librosa.filters.mel(
            sr=config.sample_rate,
            n_fft=self._N_FFT,
            n_mels=self._NUM_MELS,
            fmin=self._FMIN,
            fmax=self._FMAX,
        )

    @staticmethod
    def _normalise_state_dict(state) -> Dict[str, torch.Tensor]:
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if (
            isinstance(state, dict)
            and "model" in state
            and isinstance(state["model"], dict)
        ):
            state = state["model"]
        if not isinstance(state, dict):
            raise TypeError("Unsupported audio encoder checkpoint format")

        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("model.")
            if not key.startswith("audio_encoder."):
                key = f"audio_encoder.{key}"
            cleaned[key] = value
        return cleaned

    def _mel_spectrogram(self, audio_path: str) -> np.ndarray:
        wav, _ = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        wav = signal.lfilter([1, -self._PREEMPHASIS], [1], wav)
        spectrum = librosa.stft(
            y=wav,
            n_fft=self._N_FFT,
            hop_length=self._HOP_SIZE,
            win_length=self._WIN_SIZE,
        )
        magnitude = np.abs(spectrum)
        mel = self._mel_basis @ magnitude
        min_level = np.exp(self._MIN_LEVEL_DB / 20 * np.log(10))
        mel_db = 20 * np.log10(np.maximum(min_level, mel)) - self._REF_LEVEL_DB
        normalised = np.clip(
            (2 * self._MAX_ABS_VALUE)
            * ((mel_db - self._MIN_LEVEL_DB) / -self._MIN_LEVEL_DB)
            - self._MAX_ABS_VALUE,
            -self._MAX_ABS_VALUE,
            self._MAX_ABS_VALUE,
        )
        return normalised.T.astype(np.float32)

    def _frame_windows(self, mel: np.ndarray) -> torch.Tensor:
        mel_frames_per_second = self.config.sample_rate / self._HOP_SIZE
        usable = max(1, mel.shape[0] - self.config.mel_step_size)
        frame_count = max(
            1,
            int(usable / mel_frames_per_second * self.config.fps),
        )
        windows = []
        for frame_index in range(frame_count):
            start = int(mel_frames_per_second * frame_index / self.config.fps)
            window = mel[start : start + self.config.mel_step_size]
            if len(window) < self.config.mel_step_size:
                pad = self.config.mel_step_size - len(window)
                window = np.pad(window, ((0, pad), (0, 0)), mode="edge")
            windows.append(window.T)
        return torch.from_numpy(np.stack(windows)).unsqueeze(1).float()

    @torch.inference_mode()
    def extract(self, audio_path: str) -> np.ndarray:
        """Return one 512-D feature for every 25-fps output frame."""

        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio input not found: {audio_path}")
        windows = self._frame_windows(self._mel_spectrogram(audio_path))
        output = []
        for start in range(0, len(windows), self.config.batch_size):
            batch = windows[start : start + self.config.batch_size].to(self.device)
            output.append(self.model(batch).float().cpu())
        return torch.cat(output, dim=0).numpy().astype(np.float32)
