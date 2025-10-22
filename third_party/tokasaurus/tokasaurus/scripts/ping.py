import time

import pydra
from openai import OpenAI


class ScriptConfig(pydra.Config):
    prompt: str

    model: str = ""
    port: int = 10210
    host: str = "0.0.0.0"
    chat: bool = False
    max_tokens: int = 100
    n: int = 1
    temperature: float = 0.0
    hide: bool = False
    retries: int = 0


def ping(config: ScriptConfig):
    client = OpenAI(
        base_url=f"http://{config.host}:{config.port}/v1",
        api_key="fake-key",
        max_retries=config.retries,
    )

    print("Making request...")
    start = time.time()
    if config.chat:
        out = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": config.prompt}],
            max_tokens=config.max_tokens,
            n=config.n,
            temperature=config.temperature,
        )
        responses = [choice.message.content for choice in out.choices]
    else:
        out = client.completions.create(
            model=config.model,
            prompt=config.prompt,
            max_tokens=config.max_tokens,
            n=config.n,
            temperature=config.temperature,
        )
        responses = [choice.text for choice in out.choices]

    end = time.time()
    print(f"Time taken: {end - start} seconds")

    if not config.hide:
        print("Responses:")
        print("-" * 100)
        for i, response in enumerate(responses):
            print(f"Response {i}: {response}")
            print("-" * 100)


def main():
    pydra.run(ping)


if __name__ == "__main__":
    main()
