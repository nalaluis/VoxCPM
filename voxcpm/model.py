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

# Increased from 448 to 512 to better handle longer utterances without
# truncation — was occasionally cutting off the last few words on longer clips.
DEFAULT_MAX_NEW_TOKENS = 512


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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Raw audio waveform as a numpy array (float32, mono) or a
                path to an audio
