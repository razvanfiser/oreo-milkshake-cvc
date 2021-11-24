from datetime import datetime

def sample_responses(input_text):
    user_message = str(input_text).lower()

    if user_message in ("hello", "hi"):
        return "Hello how are you?"

    if user_message in ("who are you?"):
        return "I am a bot"

    return "I cannot understand"