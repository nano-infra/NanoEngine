import pytest
from dlengine.runtime.models.quant_config import QuantizationConfig


def test_compressed_tensors_mxfp4_is_detected():
    config = QuantizationConfig(
        quant_method="compressed-tensors",
        format="mxfp4-pack-quantized",
        config_groups={
            "group_0": {
                "weights": {
                    "group_size": 32,
                    "num_bits": 4,
                    "scale_dtype": "torch.uint8",
                }
            }
        },
    )

    assert config.is_mxfp4
    assert config.compression_format == "mxfp4-pack-quantized"
    assert config.weight_group_size == 32


def test_mxfp4_rejects_nonstandard_group_size():
    with pytest.raises(ValueError, match="group size of 32"):
        QuantizationConfig(
            quant_method="compressed-tensors",
            format="mxfp4-pack-quantized",
            config_groups={"group_0": {"weights": {"group_size": 64}}},
        )


def test_unquantized_config_is_not_mxfp4():
    config = QuantizationConfig()
    assert not config.is_mxfp4
