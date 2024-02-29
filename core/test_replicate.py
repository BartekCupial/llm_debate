import hydra
import replicate
from omegaconf import DictConfig
from transformers import AutoTokenizer

from core.utils import setup_environment


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: DictConfig):
    setup_environment(
        logger_level=cfg.logging,
        anthropic_tag=cfg.anthropic_tag,
        openai_tag=cfg.openai_tag,
        replicate_tag=cfg.replicate_tag,
    )

    # The mistralai/mixtral-8x7b-instruct-v0.1 model can stream output as it's running.
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    prompt = "Write a bedtime story about neural networks I can read to my toddler"

    # "mistralai/mixtral-8x7b-instruct-v0.1"
    response = replicate.run(
        model_id.lower(),
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": prompt,
            "temperature": 0.6,
            "max_new_tokens": 1024,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
    completion = "".join(response)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded_prompt = tokenizer(prompt)
    encoded_completion = tokenizer(completion)
    num_context_tokens, num_completion_tokens = len(encoded_prompt["input_ids"]), len(
        encoded_completion["input_ids"]
    )

    print(completion)


if __name__ == "__main__":
    main()
