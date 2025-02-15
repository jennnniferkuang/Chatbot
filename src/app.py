import tkinter as tk
from tkinter import messagebox
import threading
from chatBot import ChatBot  # Import the ChatBot class from the chatBot.py file
from tictactoe import TicTacToe

from openai import OpenAI
import os
import json
import sys
import time
import datetime

import speech_recognition as sr
from translate import Translator

from generate import load_generator, generate_face
from PIL import Image, ImageTk

generator = load_generator()

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
toneColors = {
    "indifferent": "#707070",  # Gray
    "amused": "#FFFF00",      # Yellow
    "excited": "#FFA500",     # Orange
    "pleased": "#90EE90",     # Light Green
    "confused": "#ADD8E6",    # Light Blue
    "sad": "#0000FF",         # Blue
    "worried": "#800080",     # Purple
    "frustrated": "#FF0000"   # Red
}
toneTextColors = {
    "indifferent": "white",  # Gray
    "amused": "black",      # Yellow
    "excited": "white",     # Orange
    "pleased": "white",     # Light Green
    "confused": "black",    # Light Blue
    "sad": "white",         # Blue
    "worried": "white",     # Purple
    "frustrated": "white"   # Red
}
quit_phrases = ["quit", "exit", "leave"]

# Create an instance of ChatBot
api_key = "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"
chatbot = ChatBot(api_key)


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ChatBot")

        # Set window size (larger window for better spacing)
        self.root.geometry("500x700")

        # Set background color to dark grey
        self.root.configure(bg="#2e2e2e")

        # Flag to control whether chat should continue
        self.stop_chat = False
        self.chat_thread = None

        #Title
        self.title = None

        # Language
        self.currentLanguage = "English"

        # Message history
        self.messageHistory = [{"role": "system", "content": f"You are a friendly chatbot. Your goal is to have a conversation with the user. Do not give long responses. You are to give all your responses in {self.currentLanguage} regardless of the language the user uses."}]

        # Home screen - show buttons to navigate
        self.home_screen()

        self.face_counter = 1  # Add counter for face images
        self.face_images = {}  # Store face PhotoImage objects

    def cleanup_face_images(self):
        """Delete all generated face images"""
        try:
            for i in range(1, self.face_counter):
                file_path = os.path.join('face_emotions', f'face{i}.png')
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists('face_emotions') and not os.listdir('face_emotions'):
                os.rmdir('face_emotions')
        except Exception as e:
            print(f"Error cleaning up face images: {e}")

    def on_closing(self):
        """Handle window close event"""
        self.stop_chat = True
        self.cleanup_face_images()
        self.root.destroy()

    def home_screen(self):
        self.cleanup_face_images()  # Clean up when returning to home
        # Set initial message history
        self.messageHistory = [{"role": "system", "content": f"You are a friendly chatbot. Your goal is to have a conversation with the user. Do not give long responses. You are to give all your responses in {self.currentLanguage} regardless of the language the user uses."}]
        
        # Set initial title
        self.title = None

        # Clear the window (if any widgets are present)
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a label for the home screen
        home_label = tk.Label(self.root, text="AI Chat Buddy", font=("Arial", 24, "bold"), fg="white", bg="#2e2e2e")
        home_label.pack(pady=50)

        # Create a button to go to the chatbot interface (does not start chat)
        new_chat_button = tk.Button(self.root, text="New Chat", command=lambda: self.go_to_chat_screen(), font=("Arial", 16, "bold"), bg="#555555", fg="white", width=20, height=2)
        new_chat_button.pack(pady=10)

        chat_history_button = tk.Button(self.root, text="Continue Chat", command=self.go_to_load_screen, font=("Arial", 16, "bold"), bg="#555555", fg="white", width=20, height=2)
        chat_history_button.pack(pady=10)

        go_to_tictactoe_button = tk.Button(self.root, text="Tic Tac Toe", command=self.go_to_tictactoe, font=("Arial", 16, "bold"), bg="#555555", fg="white", width=20, height=2)
        go_to_tictactoe_button.pack(pady=10)

        placeholder_button_3 = tk.Button(self.root, text="Feature 3 (To be implemented)", state=tk.DISABLED, font=("Arial", 16, "bold"), bg="#555555", fg="white", width=20, height=2)
        placeholder_button_3.pack(pady=10)

        placeholder_button_4 = tk.Button(self.root, text="Feature 4 (To be implemented)", state=tk.DISABLED, font=("Arial", 16, "bold"), bg="#555555", fg="white", width=20, height=2)
        placeholder_button_4.pack(pady=10)

        change_language_button = tk.Button(self.root, text="Language", command=self.go_to_language_screen, font=("Arial", 16, "bold"), bg="#555555", fg="white", width=20, height=2)
        change_language_button.pack(pady=10)


    def go_to_chat_screen(self):
        # Clear the home screen and start the chat interface
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the back button at the top
        back_button_frame = tk.Frame(self.root, bg="#2e2e2e", height=40)  # Added height to frame
        back_button_frame.pack(fill=tk.X, pady=(10, 20))  # Add padding between top of window and back button

        # Create a Back button in the top-left corner to go back to the home screen
        back_button = tk.Button(back_button_frame, text="Back", command=self.home_screen, font=("Arial", 12, "bold"), bg="#555555", fg="white", width=8, height=1)
        back_button.place(x=10, y=5)  # Position the button at the top-left corner with some vertical space

        # Create a text area to display the conversation
        self.text_area = tk.Text(self.root, state=tk.DISABLED, wrap=tk.WORD, bg=toneColors["indifferent"], fg="white", font=("Arial", 12))
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)  # Padding added to prevent overlap
        # Output message history
        initialText = ""
        for message in self.messageHistory:
            if(message["role"] == "user"):
                initialText += (f'User: {message["content"]}\n')
            elif(message["role"] == "assistant"):
                initialText += (f'Bot: {message["content"]}\n')
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, initialText)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

        # Create an entry field for user input
        self.entry_field = tk.Entry(self.root, state=tk.DISABLED, font=("Arial", 14))
        self.entry_field.pack(padx=10, pady=10, fill=tk.X)

        # Bind the enter key to send input
        self.entry_field.bind("<Return>", self.handle_text_input)

        # Create a button to start the conversation
        self.start_button = tk.Button(self.root, text="Start Chat", command=lambda: self.start_chat(back_button), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.start_button.pack(pady=10)
        
        self.user_icon = tk.PhotoImage(width=48, height=48)  # 48x48 box
        self.user_icon.put(("white",), to=(0, 0, 48, 48))

    def go_to_language_screen(self):
        # Clear the home screen and start the chat interface
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the back button at the top
        back_button_frame = tk.Frame(self.root, bg="#2e2e2e", height=40)  # Added height to frame
        back_button_frame.pack(fill=tk.X, pady=(10, 20))  # Add padding between top of window and back button

        # Create a Back button in the top-left corner to go back to the home screen
        back_button = tk.Button(back_button_frame, text="Back", command=self.home_screen, font=("Arial", 12, "bold"), bg="#555555", fg="white", width=8, height=1)
        back_button.place(x=10, y=5)  # Position the button at the top-left corner with some vertical space

        # Buttons for each language
        self.english_button = tk.Button(self.root, text="English", command=lambda: self.set_language("English"), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.english_button.pack(pady=10)

        self.spanish_button = tk.Button(self.root, text="Spanish", command=lambda: self.set_language("Spanish"), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.spanish_button.pack(pady=10)

        self.french_button = tk.Button(self.root, text="French", command=lambda: self.set_language("French"), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.french_button.pack(pady=10)

        self.german_button = tk.Button(self.root, text="German", command=lambda: self.set_language("German"), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.german_button.pack(pady=10)

        self.italian_button = tk.Button(self.root, text="Italian", command=lambda: self.set_language("Italian"), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.italian_button.pack(pady=10)

        self.portuguese_button = tk.Button(self.root, text="Portuguese", command=lambda: self.set_language("Portuguese"), font=("Arial", 14), bg="#555555", fg="white", width=20, height=2)
        self.portuguese_button.pack(pady=10)

        #Label indicating current language
        self.language_label = tk.Label(self.root, text=f"Current language: {self.currentLanguage}", font=("Arial", 20), fg="white", bg="#2e2e2e")
        self.language_label.pack(pady=30)
    
    def set_language(self, language):
        self.currentLanguage = language
        print(f"Language set to {self.currentLanguage}")
        self.language_label["text"] = f"Current language: {self.currentLanguage}"

    def go_to_load_screen(self):
        # Clear the home screen and start the chat interface
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the back button at the top
        back_button_frame = tk.Frame(self.root, bg="#2e2e2e", height=40)  # Added height to frame
        back_button_frame.pack(fill=tk.X, pady=(10, 20))  # Add padding between top of window and back button

        # Create a Back button in the top-left corner to go back to the home screen
        back_button = tk.Button(back_button_frame, text="Back", command=self.home_screen, font=("Arial", 12, "bold"), bg="#555555", fg="white", width=8, height=1)
        back_button.place(x=10, y=5)  # Position the button at the top-left corner with some vertical space

        if(os.path.exists("records")):
            directory = "records"
            fileList = os.listdir(directory)

            boxText = "Select a chat to load:\n"
            for i in range(len(fileList)):
                boxText += f"{i + 1}. {fileList[i][:-5]}\n"
            boxText += f"(Please enter number from 1 to {len(fileList)})\n"

            self.text_area = tk.Text(self.root, state=tk.DISABLED, wrap=tk.WORD, bg="#333333", fg="white", font=("Arial", 12))
            self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)  # Padding added to prevent overlap

            # Create an entry field for user input
            self.entry_field = tk.Entry(self.root, font=("Arial", 14))
            self.entry_field.pack(padx=10, pady=10, fill=tk.X)

            # Bind the enter key to send input
            self.entry_field.bind("<Return>", self.select_chat)

            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, boxText)
            self.text_area.config(state=tk.DISABLED)
            self.text_area.yview(tk.END)
        else:
            self.text_area = tk.Text(self.root, state=tk.DISABLED, wrap=tk.WORD, bg="#333333", fg="white", font=("Arial", 16))
            self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)  # Padding added to prevent overlap

            self.text_area.config(state=tk.NORMAL)
            self.text_area.insert(tk.END, "No chats yet")
            self.text_area.config(state=tk.DISABLED)
            self.text_area.yview(tk.END)
        

    def load_chat(self, filename):
        print(f"Loading Conversation: {filename[:-5]}")
        pathName = f"records/{filename}"
        with open(pathName, 'r') as file:
            self.messageHistory = json.load(file)
        
        self.title = filename[:-5]
        for key in languageAbbreviations.keys():
            if(key in self.messageHistory[0]["content"]):
                self.currentLanguage = key

        self.go_to_chat_screen()
    
    def select_chat(self, event=None):
        directory = "records"
        fileList = os.listdir(directory)
        

        choice = int(self.entry_field.get())
        if(choice >= 1 and choice <= len(fileList)):
            self.load_chat(fileList[choice - 1])
        else:
            self.entry_field.delete(0, tk.END)

        
       
    def handle_text_input(self, event=None):
        user_input = self.entry_field.get()

        if user_input.lower() == "quit" or user_input.lower() == "exit":
            self.stop_chat = True
            self.update_text_area("Ending chat...\n")
            return

        self.entry_field.delete(0, tk.END)
        
        # Get tone and generate face
        tone_response = chatbot.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return the tone of the following text as one of these options: angry, disgust, fear, happy, neutral, sad, surprise."},
                      {"role": "user", "content": user_input}]
        )
        tone = tone_response.choices[0].message.content.strip().lower()
        
        # Generate face
        generate_face(tone, generator, f'face{self.face_counter}.png')
        self.face_images[self.face_counter] = self.load_face_image(self.face_counter)
        current_face = self.face_counter
        self.face_counter += 1
        
        # Display user input with face
        self.update_text_area(f"You: {user_input}\n", current_face)

        # Get and display bot response 
        message_history = [{"role": "user", "content": user_input}]
        bot_response = chatbot.generateMessage(message_history)
        
        # Update text area appearance with tone
        tone = chatbot.determineTone(message_history)
        self.text_area.configure(bg=toneColors[tone], fg=toneTextColors[tone])
        
        self.update_text_area(f"ChatBot: {bot_response}\n")
        chatbot.textToSpeech(bot_response, "English", tone)

    def start_chat(self, button):
        # Remove the button.destroy() call that was deleting the back button
        # button.destroy()  <- Remove this line
        
        # Set stop_chat flag to False when chat starts
        self.stop_chat = False
        self.entry_field.config(state="normal")
        if self.chat_thread is None or not self.chat_thread.is_alive():
            # Start the chat in a new thread to keep the UI responsive
            self.chat_thread = threading.Thread(target=self.run_chat)
            self.chat_thread.daemon = True  # Ensures the thread exits when the main program exits
            self.chat_thread.start()
    
    # Function for voice input
    def run_chat(self):
        # Initialize conversation history
        message_history = []
        self.is_chatting = True

        self.update_text_area("ChatBot: I'm ready to chat!\n")

        while not self.stop_chat:
            user_input = chatbot.speechToText("English")

            if user_input.lower() == "quit" or user_input.lower() == "exit":
                self.stop_chat = True
                self.update_text_area("ChatBot: Ending chat...\n")
                break

            # Get tone and generate face before displaying input
            tone_response = chatbot.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Return the tone of the following text as one of these options: angry, disgust, fear, happy, neutral, sad, surprise."},
                         {"role": "user", "content": user_input}]
            )
            tone = tone_response.choices[0].message.content.strip().lower()
            
            # Generate face
            generate_face(tone, generator, f'face{self.face_counter}.png')
            self.face_images[self.face_counter] = self.load_face_image(self.face_counter)
            current_face = self.face_counter
            self.face_counter += 1

            # Display user input with face
            self.update_text_area(f"You: {user_input}\n", current_face)

            message_history.append({"role": "user", "content": user_input})

            # Get and display bot response
            completion = chatbot.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message_history
            )
            bot_response = completion.choices[0].message.content

            # Update text area appearance with tone
            tone = chatbot.determineTone(message_history)
            self.text_area.configure(bg=toneColors[tone], fg=toneTextColors[tone])
            
            self.update_text_area(f"ChatBot: {bot_response}\n")
            chatbot.textToSpeech(bot_response, "English", tone)
    
    # Main chat function that executes everytime an user input is received whether through text or speech
    def chat(self, userInput):
        translator = Translator(from_lang=languageAbbreviations[self.currentLanguage], to_lang="en")  
        
        # Display user input
        self.update_text_area(f"You: {userInput}\n")

        # Check if the user wants to end the chat
        if((translator.translate(userInput).lower() in quit_phrases)):
            print(self.title)
            # Record chat
            print(self.messageHistory)
            if(self.title):
                chatbot.recordChat(self.messageHistory, self.title, False)
            else:
                self.title = chatbot.generateTitle(self.messageHistory)
                chatbot.recordChat(self.messageHistory, self.title, True)
            #self.home_screen()
            return False
        
        # Add the user's message to the conversation
        self.messageHistory.append({"role": "user", "content": userInput})

        # Get the assistant's response
        bot_response = chatbot.generateMessage(self.messageHistory)
        
        # Determine the tone of the assistant - moved after getting response
        tone = chatbot.determineTone(self.messageHistory)
        
        # Update text area appearance
        self.text_area.configure(bg=toneColors[tone], fg=toneTextColors[tone])
        
        self.update_text_area(f"ChatBot: {bot_response}\n")
        # Add the assistant's response to the conversation
        self.messageHistory.append({"role": "assistant", "content": bot_response})

        # Convert the assistant's response to speech
        chatbot.textToSpeech(bot_response, self.currentLanguage, tone)

        return True

    def update_text_area(self, text, face_number=None):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, "\n")
        if text.startswith("You:"):
            if face_number and face_number in self.face_images:
                self.text_area.image_create(tk.END, image=self.face_images[face_number])
            else:
                self.text_area.image_create(tk.END, image=self.user_icon)
            self.text_area.insert(tk.END, " ")
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def delete_last_text(self):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete("end -1l", "end")
        self.text_area.insert(tk.END, "\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.yview(tk.END)

    def go_to_tictactoe(self):
        # Clear the home screen and start the chat interface
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create a frame for the back button at the top
        back_button_frame = tk.Frame(self.root, bg="#2e2e2e", height=40)  # Added height to frame
        back_button_frame.pack(fill=tk.X, pady=(10, 20))  # Add padding between top of window and back button

        # Create a Back button in the top-left corner to go back to the home screen
        back_button = tk.Button(back_button_frame, text="Back", command=self.home_screen, font=("Arial", 12, "bold"), bg="#555555", fg="white", width=8, height=1)
        back_button.place(x=10, y=5)  # Position the button at the top-left corner with some vertical space
        
        # Create a new tic tac toe game
        self.ttt_game = TicTacToe(root, self.currentLanguage)
        self.start_ttt()
    
    def run_ttt(self):
        self.ttt_game.run()
    
    def start_ttt(self):
        self.ttt_thread = threading.Thread(target=self.run_ttt)
        self.ttt_thread.daemon = True  # Ensures the thread exits when the main program exits
        self.ttt_thread.start()

    def get_bot_response(self, user_input):
        # Here, we handle the conversation logic between the user and bot
        message_history = [{"role": "user", "content": user_input}]
        
        # Send user's input to ChatGPT to get the tone
        tone_response = chatbot.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return the tone of the following text as one of these options: angry, disgust, fear, happy, neutral, sad, surprise."},
                      {"role": "user", "content": user_input}]
        )
        tone = tone_response.choices[0].message.content.strip().lower()
        
        # Generate face with numbered filename
        generate_face(tone, generator, f'face{self.face_counter}.png')
        self.face_images[self.face_counter] = self.load_face_image(self.face_counter)
        current_face = self.face_counter
        self.face_counter += 1
        
        bot_response = chatbot.generateMessage(message_history)
        
        # Update text area with user input and bot response (only once)
        self.update_text_area(f"You: {user_input}\n", current_face)
        self.update_text_area(f"ChatBot: {bot_response}\n")
        chatbot.textToSpeech(bot_response, "English", tone)

    def load_face_image(self, number):
        """Load a face image and convert it to PhotoImage"""
        try:
            image = Image.open(f'face_emotions/face{number}.png')
            image = image.resize((48, 48))  # Resize to match chat box size
            return ImageTk.PhotoImage(image)
        except Exception as e:
            print(f"Error loading face{number}.png: {e}")
            return None


# Create and run the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)

    root.mainloop()