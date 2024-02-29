import hydra
import replicate
from omegaconf import DictConfig

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
    for event in replicate.stream(
        "mistralai/mixtral-8x7b-instruct-v0.1",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": "Write a bedtime story about neural networks I can read to my toddler",
            "temperature": 0.6,
            "max_new_tokens": 1024,
            "prompt_template": "<s>[INST] {prompt} [/INST] ",
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    ):
        print(str(event), end="")


if __name__ == "__main__":
    main()
