import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

text = "This is an example sentence to count tokens."
token_count = len(encoding.encode(text))
print(f"The text contains {token_count} tokens.")
