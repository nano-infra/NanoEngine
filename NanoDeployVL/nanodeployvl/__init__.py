"""NanoDeployVL – Vision-Language inference engine for NanoInfra.

Provides EP-separated (Encoder-Prefill) VL inference pipeline where
a standalone EncoderEngine runs the vision encoder and delivers
embeddings to the LLM workers via RDMA.
"""

__version__ = "0.1.0"

from nanodeployvl.config import VLConfig
from nanodeployvl.encoder.encoder_config import EncoderConfig
from nanodeployvl.encoder.encoder_engine import EncoderEngine
from nanodeployvl.server.vl_engine_server import VLEngineServer, VLServerConfig
from nanodeployvl.vision.encoder import VisionEncoder
from nanodeployvl.vision.processor import ImageProcessor

__all__ = [
    "VLConfig",
    "EncoderConfig",
    "EncoderEngine",
    "VisionEncoder",
    "ImageProcessor",
    "VLEngineServer",  # encoder-only server (NanoRoute handles client requests)
    "VLServerConfig",
]
