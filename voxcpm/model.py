"""VoxCPM model wrapper and inference utilities.

This module provides the core model loading and inference functionality
for the VoxCPM speech recognition system.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MODEL_VERSION = "voxcpm_v1.5"


class VoxCPMModel:
    """Wrapper class for VoxCPM automatic speech recognition model.

    Handles model loading, audio preprocessing, and transcription inference.

    Args:
        model_dir (str or Path): Path to the directory containing model weights
            and configuration files.
        device (str): Device to run inference on. Defaults to 'cuda' if available,
            otherwise 'cpu'.
        dtype (torch.dtype): Data type for model weights. Defaults to float16
            on CUDA, float32 on CPU.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        # Resolve device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype
        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype

        logger.info(
            "Initializing VoxCPM model from %s on %s (%s)",
            self.model_dir,
            self.device,
            self.dtype,
        )

        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        """Load model weights and processor from the model directory."""
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_dir), trust_remote_code=True
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                str(self.model_dir),
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device)
            self._model.eval()
            logger.info("Model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        language: Optional[str] = None,
        max_new_tokens: int = 448,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw audio waveform as a numpy array (float32, mono) or a
                path to an audio file.
            sample_rate: Sample rate of the provided waveform. Ignored when
                ``audio`` is a file path.
            language: BCP-47 language tag to force the model to decode in a
                specific language (e.g. ``'zh'``, ``'en'``). If ``None``, the
                model auto-detects the language.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Transcribed text string.
        """
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio_file(audio)

        inputs = self._processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device, dtype=self.dtype)

        forced_decoder_ids = None
        if language is not None:
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=max_new_tokens,
            )

        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0].strip() if transcription else ""

    @staticmethod
    def _load_audio_file(path: Union[str, Path]):
        """Load an audio file and return (waveform, sample_rate)."""
        try:
            import soundfile as sf

            waveform, sr = sf.read(str(path), dtype="float32", always_2d=False)
            # Convert stereo to mono by averaging channels
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            return waveform, sr
        except ImportError:
            raise ImportError(
                "soundfile is required to load audio files. "
                "Install it with: pip install soundfile"
            )

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been successfully loaded."""
        return self._model is not None and self._processor is not None
