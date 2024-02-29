import hydra
import openai
from omegaconf import DictConfig
from openai import OpenAI

from core.utils import setup_environment


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: DictConfig):
    setup_environment(
        logger_level=cfg.logging,
        anthropic_tag=cfg.anthropic_tag,
        openai_tag=cfg.openai_tag,
        replicate_tag=cfg.replicate_tag,
    )
    client = OpenAI(api_key=openai.api_key)

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": "Who won the world series in 2020?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()

# WORKS! GPT3.5
