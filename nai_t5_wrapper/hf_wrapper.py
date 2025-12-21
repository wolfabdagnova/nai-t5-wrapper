"""
HuggingFace-Compatible Wrapper for NAI-T5

This module provides drop-in replacements for HuggingFace's T5EncoderModel,
MT5EncoderModel, and UMT5EncoderModel using NovelAI's optimized T5 implementation
with Flex Attention.

Supported model types:
- T5 (google/t5-v1_1-xxl, etc.)
- MT5 (google/mt5-xxl, etc.) - Multilingual T5
- UMT5 (google/umt5-xxl, etc.) - Unified Multilingual T5 with per-layer position embeddings

The wrapper:
- Downloads weights from HuggingFace and converts at runtime
- Provides HuggingFace-compatible interface (from_pretrained, forward returns BaseModelOutput-like)
- Uses Flex Attention for optimal performance (falls back to SDPA if unavailable)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import BoolTensor, FloatTensor, LongTensor, nn

logger = logging.getLogger(__name__)

# Supported model types
SUPPORTED_MODEL_TYPES = {'t5', 'mt5', 'umt5'}


@dataclass
class NAIT5EncoderOutput:
    """
    Output class mimicking HuggingFace's BaseModelOutput.
    Supports both attribute access (.last_hidden_state) and indexing ([0]).
    """

    last_hidden_state: FloatTensor
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.last_hidden_state
        raise IndexError(f"Index {idx} out of range for NAIT5EncoderOutput")

    def __iter__(self):
        yield self.last_hidden_state


class NAIT5EncoderModel(nn.Module):
    """
    HuggingFace-compatible wrapper around NovelAI's T5EncoderStack.

    This class provides a drop-in replacement for T5EncoderModel, MT5EncoderModel,
    and UMT5EncoderModel with:
    - Same from_pretrained() interface
    - Same forward() signature and return type
    - Flex Attention backend for optimal performance

    Example usage:
        from nai_t5_wrapper import NAIT5EncoderModel

        # Load any supported T5 variant
        model = NAIT5EncoderModel.from_pretrained('google/t5-v1_1-xxl')
        model = NAIT5EncoderModel.from_pretrained('google/mt5-xxl')
        model = NAIT5EncoderModel.from_pretrained('google/umt5-xxl')

        # Use like HuggingFace T5EncoderModel
        output = model(input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state  # or output[0]
    """

    _supports_flex_attention: bool = False

    def __init__(
        self,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._max_seq_len = max_seq_len
        self._dtype = dtype
        self._device = device or torch.device("cpu")
        self._encoder: Optional[nn.Module] = None
        self._config = None
        self._model_type: Optional[str] = None
        self._hf_model_id: Optional[str] = None
        self._is_compiled = False

        # Check Flex Attention availability
        self._supports_flex_attention = self._check_flex_attention_support()

    @staticmethod
    def _check_flex_attention_support() -> bool:
        """Check if PyTorch version supports Flex Attention."""
        try:
            from torch.nn.attention.flex_attention import flex_attention

            return True
        except ImportError:
            return False

    @staticmethod
    def _is_hip() -> bool:
        """Check if running on HIP/ROCm (AMD GPU)."""
        return hasattr(torch.version, 'hip') and torch.version.hip is not None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = None,
        max_seq_len: int = 512,
        subfolder: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,  # Deprecated, use dtype
        use_flex_attention: Optional[bool] = None,  # None=auto, True=force, False=disable
        **kwargs,
    ) -> "NAIT5EncoderModel":
        """
        Load NAI-T5 encoder, converting weights from HuggingFace at runtime.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID (e.g., 'google/t5-v1_1-xxl',
                'google/mt5-xxl', 'google/umt5-xxl')
            dtype: Model dtype (default: bfloat16)
            device_map: Device placement (currently uses CUDA if available)
            max_seq_len: Maximum sequence length for Flex Attention (default: 512)
            subfolder: Subfolder within model repo (passed to HF, typically ignored)
            torch_dtype: Deprecated, use dtype instead
            use_flex_attention: Control Flex Attention usage (None=auto, True=force, False=disable).
                For UMT5 on HIP/ROCm, Flex Attention is automatically disabled due to Triton compatibility
                issues (HSA_STATUS_ERROR_EXCEPTION crashes). Set to True to force-enable anyway.
            **kwargs: Additional arguments (ignored)

        Returns:
            NAIT5EncoderModel instance with loaded weights
        """
        # Handle deprecated torch_dtype parameter
        if torch_dtype is not None:
            import warnings
            warnings.warn(
                "`torch_dtype` is deprecated! Use `dtype` instead!",
                DeprecationWarning,
                stacklevel=2,
            )
            dtype = torch_dtype

        # Determine device
        if device_map == "auto" or device_map is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_map)

        # Create instance
        instance = cls(max_seq_len=max_seq_len, dtype=dtype, device=device)
        instance._hf_model_id = pretrained_model_name_or_path

        # Override Flex Attention detection if explicitly specified
        if use_flex_attention is not None:
            instance._supports_flex_attention = use_flex_attention

        # Load the model (may disable Flex Attention for UMT5 on HIP)
        instance._load_model(pretrained_model_name_or_path, subfolder, use_flex_attention)

        return instance

    def _load_model(
        self, model_id: str, subfolder: Optional[str] = None, use_flex_attention: Optional[bool] = None
    ):
        """Load and convert HuggingFace T5/MT5/UMT5 weights to NAI-T5 format."""
        from nai_t5_wrapper.t5_encoder import T5EncoderStack
        from nai_t5_wrapper.t5_hf import hf_to_based_t5_enc_state, to_based_config, SUPPORTED_MODEL_TYPES
        from nai_t5_wrapper.t5_common import T5AttnImpl
        from nai_t5_wrapper.fuse_norm_scales import fuse_norm_scales_enc

        from transformers import AutoConfig

        logger.info(f"Loading NAI-T5 encoder from {model_id}...")

        # Get HuggingFace config to determine model type
        hf_config = AutoConfig.from_pretrained(model_id)
        self._model_type = getattr(hf_config, 'model_type', 't5')

        if self._model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{self._model_type}'. "
                f"Supported types: {SUPPORTED_MODEL_TYPES}"
            )

        logger.info(f"Detected model type: {self._model_type}")

        # Auto-disable Flex Attention for UMT5 on HIP/ROCm due to Triton compatibility issues
        if self._model_type == 'umt5' and self._supports_flex_attention and self._is_hip():
            if use_flex_attention is True:
                logger.warning(
                    "UMT5 with Flex Attention on HIP/ROCm is unstable and may crash. "
                    "You explicitly requested use_flex_attention=True, proceeding anyway."
                )
            else:
                logger.warning(
                    "Disabling Flex Attention for UMT5 on HIP/ROCm due to Triton compatibility issues. "
                    "Falling back to SDPA."
                )
                self._supports_flex_attention = False

        # Build NAI-T5 config from HuggingFace config
        attn_impl = T5AttnImpl.Flex if self._supports_flex_attention else T5AttnImpl.SDPA
        if not self._supports_flex_attention:
            logger.warning(
                "Flex Attention not available (requires PyTorch 2.5+). "
                "Falling back to SDPA. Performance may be reduced."
            )

        # Convert HF config to NAI-T5 config
        self._config = to_based_config(hf_config, n_tokens=self._max_seq_len)
        self._config = self._config.model_copy(update={
            'emb_weight_dtype': self._dtype,
            'linear_weight_dtype': self._dtype,
            'norm_weight_dtype': self._dtype,
            'attn_impl': attn_impl,
            'elementwise_affine': True,  # Will be set to False after fusion
            'flex_kernel_options': {'BLOCK_M': 128, 'BLOCK_N': 64} if self._supports_flex_attention else {},
        })

        # Load HuggingFace encoder model
        logger.info(f"Downloading and loading HuggingFace weights from {model_id}...")
        hf_encoder = self._load_hf_encoder(model_id, self._model_type)

        # Create encoder on device
        logger.info("Creating NAI-T5 encoder structure...")
        self._encoder = T5EncoderStack(self._config).to(device=self._device, dtype=self._dtype)

        # Convert state dict
        logger.info("Converting HuggingFace weights to NAI-T5 format...")
        hf_state = hf_encoder.state_dict()
        nai_state = hf_to_based_t5_enc_state(hf_state, self._config)

        # Load weights
        self._encoder.load_state_dict(nai_state)
        self._encoder.eval()

        # Free HF model memory
        del hf_encoder, hf_state, nai_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fuse norm scales for better performance
        logger.info("Fusing norm scales...")
        fuse_norm_scales_enc(self._encoder, fuse_via_f32=True)

        # Bind score mods for Flex Attention
        if self._supports_flex_attention:
            logger.info(f"Binding Flex Attention score mods for seq_len={self._max_seq_len}...")
            with torch.inference_mode():
                self._encoder.bind_score_mods(seq_len=self._max_seq_len)

        logger.info(f"NAI-T5 encoder loaded successfully ({self._model_type}).")

    def _load_hf_encoder(self, model_id: str, model_type: str):
        """Load the appropriate HuggingFace encoder model based on model type."""
        if model_type == 'umt5':
            from transformers import UMT5EncoderModel
            return UMT5EncoderModel.from_pretrained(model_id, torch_dtype=self._dtype)
        elif model_type == 'mt5':
            from transformers import MT5EncoderModel
            return MT5EncoderModel.from_pretrained(model_id, torch_dtype=self._dtype)
        else:  # t5
            from transformers import T5EncoderModel
            return T5EncoderModel.from_pretrained(model_id, torch_dtype=self._dtype)

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> NAIT5EncoderOutput:
        """
        Forward pass matching HuggingFace T5EncoderModel signature.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (1 for real tokens, 0 for padding)
            head_mask: Ignored (not supported by NAI-T5)
            inputs_embeds: Ignored (not supported by NAI-T5)
            output_attentions: Ignored (not supported by NAI-T5)
            output_hidden_states: Ignored (not supported by NAI-T5)
            return_dict: Ignored (always returns NAIT5EncoderOutput)

        Returns:
            NAIT5EncoderOutput with last_hidden_state
        """
        if self._encoder is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        # Convert attention_mask to BoolTensor (NAI-T5 expects True for real tokens)
        input_mask: Optional[BoolTensor] = None
        if attention_mask is not None:
            input_mask = attention_mask.bool()

        # Forward through NAI-T5 encoder
        # Use no_grad since NAI-T5 is designed for inference, and flex attention
        # has issues with inference_mode() (creates tensors that can't be saved for backward)
        with torch.no_grad():
            if self._supports_flex_attention and input_mask is not None:
                from nai_t5_wrapper.t5_encoder import make_self_attn_block_mask
                from torch.nn.attention.flex_attention import create_block_mask

                # Use non-compiled create_block_mask for broader compatibility
                block_mask = make_self_attn_block_mask(
                    mask=input_mask,
                    mask_pad_queries=True,
                    create_block_mask=create_block_mask,
                )
                hidden_states = self._encoder(
                    input_ids=input_ids,
                    block_mask=block_mask,
                )
            else:
                hidden_states = self._encoder(
                    input_ids=input_ids,
                    input_mask=input_mask,
                )

        return NAIT5EncoderOutput(last_hidden_state=hidden_states)

    def compile(self, **kwargs) -> "NAIT5EncoderModel":
        """
        Compile the encoder with torch.compile for additional speedup.

        Args:
            **kwargs: Arguments passed to torch.compile

        Returns:
            self for method chaining
        """
        if self._encoder is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        if not self._is_compiled:
            logger.info("Compiling NAI-T5 encoder with torch.compile...")
            compile_kwargs = {"dynamic": False, "fullgraph": True}
            compile_kwargs.update(kwargs)
            self._encoder = torch.compile(self._encoder, **compile_kwargs)
            self._is_compiled = True

        return self

    def to(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "NAIT5EncoderModel":
        """Move model to device/dtype."""
        if device is not None:
            self._device = torch.device(device) if isinstance(device, str) else device
        if dtype is not None:
            self._dtype = dtype

        if self._encoder is not None:
            self._encoder = self._encoder.to(device=self._device, dtype=self._dtype)

        return self

    def eval(self) -> "NAIT5EncoderModel":
        """Set model to evaluation mode."""
        if self._encoder is not None:
            self._encoder.eval()
        return self

    def train(self, mode: bool = True) -> "NAIT5EncoderModel":
        """Set model to training mode (NAI-T5 is inference-only, so this is a no-op)."""
        if mode:
            logger.warning("NAI-T5 is designed for inference only. Training mode has no effect.")
        return self

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model."""
        return self._dtype

    @property
    def config(self):
        """Return a config-like object for compatibility."""
        return self._config

    @property
    def model_type(self) -> Optional[str]:
        """Return the model type (t5, mt5, or umt5)."""
        return self._model_type

    def requires_grad_(self, requires_grad: bool = True) -> "NAIT5EncoderModel":
        """Set requires_grad for all parameters."""
        if self._encoder is not None:
            for param in self._encoder.parameters():
                param.requires_grad_(requires_grad)
        return self

    def parameters(self):
        """Return model parameters."""
        if self._encoder is not None:
            return self._encoder.parameters()
        return iter([])

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return named parameters."""
        if self._encoder is not None:
            return self._encoder.named_parameters(prefix=prefix, recurse=recurse)
        return iter([])

    def state_dict(self, *args, **kwargs):
        """Return state dict."""
        if self._encoder is not None:
            return self._encoder.state_dict(*args, **kwargs)
        return {}
