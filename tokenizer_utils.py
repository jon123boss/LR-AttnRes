# tokenizer_utils.py

GPT4_TOKENIZER_MODEL = "gpt-4"
GPT4_ENCODING_NAME = "cl100k_base"
GPT4_VOCAB_SIZE = 100277
GPT4_EOT_TOKEN = 100257


def get_tiktoken_encoding(tokenizer_model: str = GPT4_TOKENIZER_MODEL):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(tokenizer_model)
    except KeyError:
        if tokenizer_model == GPT4_TOKENIZER_MODEL:
            return tiktoken.get_encoding(GPT4_ENCODING_NAME)
        raise


def tokenizer_metadata(tokenizer_model: str = GPT4_TOKENIZER_MODEL):
    enc = get_tiktoken_encoding(tokenizer_model)
    return {
        "tokenizer_model": tokenizer_model,
        "encoding_name": enc.name,
        "vocab_size": int(enc.n_vocab),
        "eot_token": int(enc.eot_token),
    }
