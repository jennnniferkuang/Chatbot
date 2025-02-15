from openai import OpenAI
import os
import json
import sys
import time
import datetime

import speech_recognition as sr
from translate import Translator

defaultLanguage = "English"
languageAbbreviations = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt"
}
toneEffects = {
    "indifferent": '-a85 -p40 -s150 -g10',
    "amused": '-a120 -p70 -s200 -g0',
    "excited": '-a140 -p85 -s225 -g0',
    "pleased": '-a110 -p60 -s175 -g0',
    "confused": '-a90 -p50 -s150 -g15',
    "sad": '-a70 -p30 -s140 -g30',
    "worried": '-a100 -p65 -s190 -g10',
    "frustrated": '-a150 -p25 -s180 -g0'
}
quit_phrases = ["quit", "exit", "leave"]

class ChatBot:
    def __init__(self, key):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=key)

    # Convert speech to text
    def speechToText(self, lang):        
        said = ""
        while(said == ""):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                sys.stdout.write("You: ")
                sys.stdout.flush()
                audio = r.listen(source, phrase_time_limit=3)
                said = ""
                try:
                    said = r.recognize_google(audio, language=languageAbbreviations[lang])
                    print(said)
                except Exception as e:
                    print("\nPlease try again.")
                    time.sleep(0.5)
        
        return said

    # Convert text to speech
    def textToSpeech(self, response, language, tone):
        if(tone not in toneEffects.keys()):
            tone = "indifferent"
        command = f'espeak -v{languageAbbreviations[language]} {toneEffects[tone]} "{response}"'
        os.system(command)

    # Store message history in a json file
    def recordChat(self, messageHistory, title, newChat):
        #Store in json file
        folder = "records"
        if not os.path.exists(folder):
            os.makedirs(folder)
        pathName = f"{folder}/{title}.json"
        with open(pathName, "w") as file:
            json.dump(messageHistory, file, indent=4)

        #Store in txt file
        folder = "transcripts"
        if not os.path.exists(folder):
            os.makedirs(folder)
        pathName = f"{folder}/{title}.txt"
        with open(pathName, "w") as file:
            #Add date in transcript
            timestamp = datetime.datetime.now().strftime("%m/%d/%Y")
            file.write(f"Title: {title}\n")
            file.write(f"Date: {timestamp}\n")
            file.write("\n")
            for message in messageHistory:
                if(message["role"] == "user"):
                    file.write(f'User: {message["content"]}\n')
                elif(message["role"] == "assistant"):
                    file.write(f'Bot: {message["content"]}\n')
    
        if(newChat):
            print(f'Chat saved as "{title}"')
        else:
            print("Chat saved")

    # Load a previous conversation stored in a json file
    def loadChat(self, filename):
        print(f"Loading Conversation: {filename[:-5]}")
        pathName = f"records/{filename}"
        with open(pathName, 'r') as file:
            messageHistory = json.load(file)

        # Output message history
        for message in messageHistory:
            if(message["role"] == "user"):
                print(f'User: {message["content"]}')
            elif(message["role"] == "assistant"):
                print(f'Bot: {message["content"]}')

        language = "English"
        for key in languageAbbreviations.keys():
            if(key in messageHistory[0]["content"]):
                language = key

        # Enable user to continue conversation
        self.chat(messageHistory, filename[:-4], language)

    def displayChat(self, filename):
        file = open(f"transcripts/{filename}", "r")
        print()
        for line in file:
            print(line, end = "")

    def generateMessage(self, messageHistory):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages = messageHistory
        )
        response = completion.choices[0].message.content
        return response

    def generateTitle(self, messageHistory):
        messageHistory.append({"role": "user", "content": "This is the end of our chat. If you were to come up with a brief title for our chat (no longer than 50 characters and no punctuation marks), what would it be?"})
        title = self.generateMessage(messageHistory)
        messageHistory.pop()
        return title

    def determineTone(self, messageHistory):
        messageHistory.append({"role": "user", "content": f"You are to describe your current tone after the user's last message. How are you feeling right now. Select a tone word from the following list: {' ,'.join(toneEffects.keys())}. Keep your response limited to only one word from the list without any punctuation marks and in ENGLISH!. Again, make sure your response is in English (disregard the first message)."})
        tone = self.generateMessage(messageHistory)
        messageHistory.pop()
        print(tone.lower())
        return tone.lower()

    # Start chat with bot
    def chat(self, messageHistory, title, language):
        # Adjust for background noise
        with sr.Microphone() as source:
            r = sr.Recognizer()
            r.adjust_for_ambient_noise(source, duration=3)
            print("Adjust for background noise")

        # Initalize the Translator object
        print(language)
        translator = Translator(from_lang=languageAbbreviations[language], to_lang="en")  

        while True:
            # Get user input
            #userInput = self.speechToText(language)
            userInput = input("User: ")

            # Check if the user wants to end the chat
            if((translator.translate(userInput).lower() in quit_phrases)):
                # Record chat
                if(title):
                    self.recordChat(messageHistory, title, False)
                else:
                    title = self.generateTitle(messageHistory)
                    self.recordChat(messageHistory, title, True)
                break
            
            # Add the user's message to the conversation
            messageHistory.append({"role": "user", "content": userInput})

            # Determine the tone of the assistant
            tone = self.determineTone(messageHistory)

            # Get the assistant's response
            assistant_response = self.generateMessage(messageHistory)

            print(f"Bot: {assistant_response}")

            # Add the assistant's response to the conversation
            messageHistory.append({"role": "assistant", "content": assistant_response})

            # Convert the assistant's response to speech
            self.textToSpeech(assistant_response, language, tone)

# Main function
def main():
    api_key = "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"
    chatbot = ChatBot(api_key)
    
    language = defaultLanguage

    while(True):
        print("What do you want to do? (new / load / language / history / quit)")
        response = input().lower()
        
        if response == "new":
            #Initialize conversation
            messageHistory = [{"role": "system", "content": f"You are a friendly chatbot. Your goal is to have a conversation with the user. Do not give long responses. You are to give all your responses in {language} regardless of the language the user uses."}]
            chatbot.chat(messageHistory, None, language)

        elif response == "load":
            directory = "records"
            fileList = os.listdir(directory)
            print("Select a chat to load:")

            for i in range(len(fileList)):
                print(f"{i + 1}. {fileList[i][:-5]}")

            choice = int(input("Enter choice: "))
            while(choice < 1 or choice > len(directory)):
                choice = int(input(f"Please enter number from 1 to {len(fileList)}: "))
            chatbot.loadChat(fileList[choice - 1])

        elif response == "language":
            print("List of Available Languages:")
            for key in languageAbbreviations.keys():
                print(f"- {key}")
            language = ""
            while language not in languageAbbreviations.keys():
                language = input("Select a Language: ").lower().capitalize()
            print(f"Language set to {language}")

        elif response == "history":
            directory = "transcripts"
            fileList = os.listdir(directory)
            print("Select a chat to display:")

            for i in range(len(fileList)):
                print(f"{i + 1}. {fileList[i][:-5]}")

            choice = int(input("Enter choice: "))
            while(choice < 1 or choice > len(directory)):
                choice = int(input(f"Please enter number from 1 to {len(fileList)}: "))
            chatbot.displayChat(fileList[choice - 1])

        elif response == "quit" or response == "exit":
            break

        else:
            print("Invalid response. Please try again.")
        print()

#main()