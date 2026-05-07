from common import CharTokenizer

tok = CharTokenizer("User: Hello\nBot: Hi\n")
history = [{"user": "Hello", "bot": "Hi"}]
prompt = "Hello"
text = "\n".join(
    [f"User: {h['user']}\nBot: {h['bot']}" for h in history]
    + [f"User: {prompt}", "Bot:"]
)
ids = tok.encode(text)
print(len(ids) > 0)
