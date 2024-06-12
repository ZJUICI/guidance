import logging
import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

    from vllm import SamplingParams
    from vllm.outputs import RequestOutput

import torch
from transformers import AutoConfig, AutoTokenizer


logger = logging.getLogger("__guidance__display__")


class TGIModel:
    def __init__(self, config: PretrainedConfig, endpoint=None) -> None:
        from text_generation import AsyncClient

        self.config = config
        self.client = AsyncClient(
            endpoint
            if endpoint
            else os.getenv("TGI_ENDPOINT_URL", "http://127.0.0.1:8080")
        )

    async def generate(self, prompt: str, **kwargs):
        logger.debug(f"input prompt :{prompt},kwargs:{kwargs}")
        text = await self.client.generate(prompt, **kwargs)

        # TODO: maybe? use other method? because this is only for pydantic 2
        res_dict: dict = text.model_dump()
        res_text = res_dict["generated_text"]

        logger.debug(f"res_text: {res_text}")
        for stop_word in kwargs.get("stop_sequences", []):
            if stop_word in res_text:
                res_dict["generated_text"] = res_text[: res_text.index(stop_word)]

        return res_dict


class VllmServiceModel:
    def __init__(self, model_name_or_path: str, endpoint=None) -> None:
        import httpx

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.endpoint = (
            endpoint
            if endpoint
            else os.getenv("VLLM_ENDPOINT_URL", "http://127.0.0.1:8080")
        )

    def generate(self, request_dict: dict, **kwargs):
        logger.debug(f"request dic:{request_dict}")
        prompt_ids_length = len(request_dict["prompt_token_ids"])
        # TODO: use httpx instead of the requests
        with httpx.Client(base_url=self.endpoint) as client:
            ret = client.post("/generate", json=request_dict).json()
        # ret = requests.post(self.endpoint + "/generate", json=request_dict).json()

        # self.all_time += e - s

        sequences_id_len = len(ret["sequences"][0][prompt_ids_length:])
        self.new_tokens_count += sequences_id_len

        logger.debug(f"vllm svc return json:{ret}")
        return ret


class VllmModel:
    def __init__(self, model_name_or_path: str) -> None:
        from vllm import LLM

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = LLM(model=model_name_or_path)
        self.tokenizer = self.model.get_tokenizer()
        # self.all_time = 0
        # self.new_tokens_count = 0

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: "SamplingParams",
        **kwargs,
    ):
        logging.debug(f"input token ids: {prompt_token_ids}")

        # prompt_ids_length = len(prompt_token_ids)
        # s = time.time()

        outputs: "RequestOutput" = self.model.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # e = time.time()
        # self.all_time += e - s

        output_token_ids: list = outputs[0].outputs[0].token_ids

        # sequences_id_len = len(output_token_ids[prompt_ids_length:])
        self.new_tokens_count += len(output_token_ids)

        logging.debug(f"output token ids: {output_token_ids}")

        return torch.as_tensor([prompt_token_ids[0] + output_token_ids])


## see https://github.com/triton-inference-server/tensorrtllm_backend/blob/release/0.5.0/inflight_batcher_llm/client/inflight_batcher_llm_client.py
class TrtLLMTritonServerModel:
    pass
