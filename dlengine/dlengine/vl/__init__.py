"""dlengine.vl – Vision-Language inference engine for NanoInfra.

Provides EP-separated (Encoder-Prefill) VL inference pipeline where
a standalone EncoderEngine runs the vision encoder and delivers
embeddings to the LLM workers via RDMA.
"""

__version__ = "0.1.0"

from dlengine.vl.config import VLConfig
from dlengine.vl.encoder.encoder_config import EncoderConfig
from dlengine.vl.encoder.encoder_engine import EncoderEngine
from dlengine.vl.server.vl_engine_server import VLEngineServer, VLServerConfig
from dlengine.vl.vision.encoder import VisionEncoder
from dlengine.vl.vision.processor import ImageProcessor

__all__ = [
    "VLConfig",
    "EncoderConfig",
    "EncoderEngine",
    "VisionEncoder",
    "ImageProcessor",
    "VLEngineServer",  # encoder-only server (dlengine-router handles client requests)
    "VLServerConfig",
]
