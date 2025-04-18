import multiprocessing
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Protocol

import click
import numpy as np
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
from numpy.typing import NDArray


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class CanarySTT:
    """
    Implements the FastRTC STTModel protocol.

    Attributes:
        model_id: The Hugging Face model ID
        device: The device to run inference on ('cpu', 'cuda', 'mps')
        dtype: Data type for model weights (float16, float32)
    """

    MODEL_OPTIONS = Literal["nvidia/canary-1b", "nvidia/canary-1b-flash"]

    def __init__(
        self,
        model: MODEL_OPTIONS = "nvidia/canary-1b-flash",
        device: Optional[str] = None,
        dtype: Literal["float16", "float32"] = "float16",
    ):
        """
        Initialize the Distil-Whisper STT model.

        Args:
            model: Model size/variant to use
            device: Device to use for inference (auto-detected if None)
            dtype: Model precision (float16 recommended for faster inference)
        """
        self.model_id = model

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if dtype == "float16" else torch.float32

        # Load the model
        self._load_model(model=model, lang="en", beam_size=1)

    def _load_model(self, model: str, lang: str, beam_size: int):
        try:
            self.lang = lang
            self.beam_size = beam_size
            self._suppress_nemo_warnings()
            self.model = EncDecMultiTaskModel.from_pretrained(model)
            decode_cfg = self.model.cfg.decoding
            decode_cfg.beam.beam_size = self.beam_size
            self.model.change_decoding_strategy(decode_cfg)
            self._setup_transcribe_dataloader()
            self._monkey_patch_nemo_warnings()
        except (ImportError, ModuleNotFoundError):
            raise ImportError("Install nemo_toolkit[asr] to use the Canary STT model.")

    def _setup_transcribe_dataloader(self):
        """
            Configure the model to minimize warnings during transcription.
            The dataloader with trim_silence is not supported by the Canary model.
        """
        try:
            import nemo.collections.common.data.lhotse.dataloader as lhotse_loader
            from nemo.collections.common.data.lhotse.dataloader import (
                make_structured_with_schema_warnings,
            )
            from omegaconf import OmegaConf

            original_make_structured = make_structured_with_schema_warnings

            def patched_make_structured(config):
                config_copy = OmegaConf.to_container(config, resolve=True)
                config_copy.pop("trim_silence", None)
                config = OmegaConf.create(config_copy)
                config.num_workers = 0
                return original_make_structured(config)

            lhotse_loader.make_structured_with_schema_warnings = patched_make_structured
        except Exception as e:
            print(click.style("WARN", fg="yellow") + f":\t  Could not patch dataloader: {str(e)}")
            pass

    def _monkey_patch_nemo_warnings(self):
        """
            Function to directly patch the specific warning in NeMo's dataloader.py
            This is a workaround to avoid the warning about the non-tarred dataset and requested tokenization.
        """
        import importlib.util
        import logging
        if importlib.util.find_spec("nemo.collections.common.data.lhotse.dataloader"):
            try:
                nemo_logger = logging.getLogger('nemo')

                class VerySpecificFilter(logging.Filter):
                    def filter(self, record):
                        if hasattr(record, 'msg') and isinstance(record.msg, str):
                            if "non-tarred dataset and requested tokenization" in record.msg:
                                return False
                        return True
                nemo_logger.addFilter(VerySpecificFilter())

                for name in logging.root.manager.loggerDict:
                    if name.startswith('nemo'):
                        logging.getLogger(name).addFilter(VerySpecificFilter())

                if importlib.util.find_spec("nemo.collections.common.data.lhotse.dataloader"):
                    import nemo.collections.common.data.lhotse.dataloader as dataloader_module
                    if hasattr(dataloader_module, 'get_lhotse_dataloader_from_config'):
                        original_func = dataloader_module.get_lhotse_dataloader_from_config

                        def wrapped_func(*args, **kwargs):
                            for handler in logging.root.handlers:
                                handler.addFilter(VerySpecificFilter())
                            result = original_func(*args, **kwargs)
                            for handler in logging.root.handlers:
                                try:
                                    handler.removeFilter(VerySpecificFilter())
                                except:
                                    pass
                            return result
                        dataloader_module.get_lhotse_dataloader_from_config = wrapped_func
                return True
            except Exception as e:
                print(f"Failed to patch NeMo warnings: {str(e)}")
                return False
        return False

    def _suppress_nemo_warnings(self):
        """
            Suppress specific loggers with special focus on dataloader
        """
        import logging
        for logger_name in ['nemo', 'nemo.collections', 'nemo.collections.asr', 'nemo_logger']:
            try:
                logging.getLogger(logger_name).setLevel(logging.ERROR)
            except:
                pass

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """
        Transcribe audio to text using distil-whisper.

        Args:
            audio: Tuple of (sample_rate, audio_data)
                  where audio_data is a numpy array of int16 or float32

        Returns:
            Transcribed text as string
        """
        sample_rate, audio_np = audio
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Handle different audio formats
        if audio_np.dtype == np.int16:
            # Convert int16 to float32 and normalize to [-1, 1]
            audio_np = audio_np.astype(np.float32) / 32768.0

        try:
            import tempfile

            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                sf.write(temp_file.name, audio_np, 16000)
                result = self.model.transcribe(
                    temp_file.name,
                    source_lang=self.lang,
                    target_lang=self.lang,
                    task='asr'
                )[0]
                return result.text
        except Exception as e:
            print(click.style("ERROR", fg="red") + f":\t  Transcription failed: {str(e)}")
            return ""


# For simpler imports
@lru_cache
def get_stt_model(
    model_name: str = "nvidia/canary-1b-flash",
    verbose: bool = True,
    **kwargs
) -> STTModel:
    """
    Helper function to easily get an STT model instance with warm-up.

    Args:
        model_name: Name of the model to use
        verbose: Whether to print status messages
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        A warmed-up STTModel instance
    """
    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create the model - remove verbose from kwargs to avoid TypeError
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'verbose'}
    m = CanarySTT(model=model_name, **filtered_kwargs)

    # Warm up the model
    curr_dir = Path(__file__).parent

    # Load test audio file for warm-up
    import numpy as np
    sample_rate = 16000

    # Try to load test_file.wav if it exists, otherwise use silence
    test_file_path = curr_dir / "test_file.wav"
    if test_file_path.exists():
        # Simple load of the test file
        # This is a placeholder - in a real implementation, you'd use a proper audio loading function
        audio = np.zeros(sample_rate, dtype=np.float32)  # Placeholder
    else:
        # Create a simple silence array for warm-up
        audio = np.zeros(sample_rate, dtype=np.float32)  # 1 second of silence

    # Print only to stderr with green styling
    if verbose:
        msg = click.style("INFO", fg="green") + ":\t  Warming up STT model.\n"
        sys.stderr.write(msg)
        sys.stderr.flush()

    # Warm up the model
    m.stt((sample_rate, audio))

    if verbose:
        msg = click.style("INFO", fg="green") + ":\t  STT model warmed up.\n"
        sys.stderr.write(msg)
        sys.stderr.flush()

    return m


if __name__ == "__main__":
    multiprocessing.freeze_support()
