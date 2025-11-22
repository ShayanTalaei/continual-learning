import requests

# Make a request with a cartridge from HuggingFace
response = requests.post("http://localhost:10210/custom/cartridge/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Help me understand this."}],
    "max_tokens": 50,
    "cartridges": [{
        "id": "pepsi_10k",
        "source": "local",
    }]
    # "cartridges": [{
    #     "id": "hazyresearch/cartridge-wauoq23f",
    #     "source": "huggingface",
    #     "force_redownload": False
    # }]
})

print(response.json())