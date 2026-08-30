from dlengine.server.openai_server import OpenAIServer


def test_request_model_alias_is_case_insensitive():
    server = OpenAIServer.__new__(OpenAIServer)
    server.served_model_name = "GLM-5.2-NVFP4"
    server._model_aliases = {
        "GLM-5.2-NVFP4",
        "/hgpfs/models/GLM-5.2-NVFP4",
    }

    assert server.resolve_request_model("glm-5.2-nvfp4") == "GLM-5.2-NVFP4"
    assert (
        server.resolve_request_model("/hgpfs/models/glm-5.2-nvfp4")
        == "GLM-5.2-NVFP4"
    )
