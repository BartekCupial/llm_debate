import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from traceback import format_exc
from typing import Dict, Optional, Union

import attrs
import replicate
from termcolor import cprint
from transformers import AutoTokenizer

from core.llm_api.base_llm import (
    PRINT_COLORS,
    LLMResponse,
    ModelAPIProtocol,
    messages_to_single_prompt,
)
from core.llm_api.openai_llm import OAIChatPrompt

MISTRAL_MODELS = {"mistralai/Mixtral-8x7B-Instruct-v0.1"}
LOGGER = logging.getLogger(__name__)


def count_tokens(prompt: str) -> int:
    return len(prompt.split())


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for the given model id.
    """
    return 0, 0


@attrs.define()
class MistralChatModel(ModelAPIProtocol):
    print_prompt_and_response: bool = False

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, response):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(response.to_dict(), indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OAIChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert (
            len(model_ids) == 1
        ), "Mistral implementation only supports one model at a time."
        model_id = model_ids[0]
        if isinstance(prompt, list):
            prompt = messages_to_single_prompt(prompt)

        prompt_file = self._create_prompt_history_file(prompt)
        LOGGER.debug(f"Making {model_id} call")
        response: Optional[Dict] = None
        duration = None
        for i in range(max_attempts):
            try:
                api_start = time.time()
                response = replicate.run(
                    model_id.lower(),
                    input={
                        "prompt": prompt,
                        "prompt_template": "<s>[INST] {prompt} [/INST] ",
                        **kwargs,
                    },
                )
                api_duration = time.time() - api_start
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(
                    f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if response is None:
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts."
            )

        completion = "".join(response)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encoded_prompt = tokenizer(prompt)
        encoded_completion = tokenizer(completion)
        num_context_tokens, num_completion_tokens = len(
            encoded_prompt["input_ids"]
        ), len(encoded_completion["input_ids"])
        context_token_cost, completion_token_cost = price_per_token(model_id)
        cost = (
            num_context_tokens * context_token_cost
            + num_completion_tokens * completion_token_cost
        )
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        stop_reason = (
            "stop" if num_completion_tokens < kwargs["max_new_tokens"] + 2 else "length"
        )

        llm_response = LLMResponse(
            model_id=model_id,
            completion=completion,
            stop_reason=stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=cost,
        )

        self._add_response_to_prompt_file(prompt_file, llm_response)
        if self.print_prompt_and_response or print_prompt_and_response:
            cprint(prompt, "yellow")
            pattern = r"(Human: |Assistant: )(.*?)(?=(Human: |Assistant: )|$)"
            for match in re.finditer(
                pattern, prompt, re.S
            ):  # re.S makes . match any character, including a newline
                role = match.group(1).removesuffix(": ").lower()
                role = {"human": "user"}.get(role, role)
                cprint(match.group(2), PRINT_COLORS[role])
            cprint(f"Response ({llm_response.model_id}):", "white")
            cprint(
                f"{llm_response.completion}", PRINT_COLORS["assistant"], attrs=["bold"]
            )
            print()

        return [llm_response]
