from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key="YOUR_KEY")
speech_file_path = Path(__file__).parent / "speech.mp3"
log_file_path = Path(__file__).parent / "logs.txt"

def text_interpreter():
    sentence=input()
    interpret=input()
    # Tell me the tone of this sentence in one word without a period
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"{interpret}: {sentence}"
            }
        ]   
    )
    res = completion.choices[0].message.content
    print(res)

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{sentence} ({interpret}): {res}\n")

def speech_file():
    sentence = input()
    response = client.audio.speech.create(
        model = "tts-1",
        voice = "alloy", #change voice
        input = sentence
    )
    response.stream_to_file(speech_file_path)

text_interpreter()
