"""nanodeploy.vl – Vision-Language inference engine for NanoInfra.

Provides EP-separated (Encoder-Prefill) VL inference pipeline where
a standalone EncoderEngine runs the vision encoder and delivers
embeddings to the LLM workers via RDMA.
"""

__version__ = "0.1.0"

from nanodeploy.vl.config import VLConfig
from nanodeploy.vl.encoder.encoder_config import EncoderConfig
from nanodeploy.vl.encoder.encoder_engine import EncoderEngine
from nanodeploy.vl.server.vl_engine_server import VLEngineServer, VLServerConfig
from nanodeploy.vl.vision.encoder import VisionEncoder
from nanodeploy.vl.vision.processor import ImageProcessor

__all__ = [
    "VLConfig",
    "EncoderConfig",
    "EncoderEngine",
    "VisionEncoder",
    "ImageProcessor",
    "VLEngineServer",  # encoder-only server (NanoRoute handles client requests)
    "VLServerConfig",
]
