import tkinter as tk
from random import randint
from openai import OpenAI
import speech_recognition as sr
import time
import sys

CANVAS_SIZE = 450
CELL_SIZE = CANVAS_SIZE // 3
INF = float('inf')

defaultLanguage = "English"
languageAbbreviations = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt"
}

class TicTacToe:
    def __init__(self, root, language):
        self.root = root
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()
        self.game_array = [[INF, INF, INF], 
                           [INF, INF, INF], 
                           [INF, INF, INF]]
        self.player = 1  # 1 for Player, 0 for AI
        self.winner = None
        self.game_steps = 0
        self.client = OpenAI(api_key="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")
        self.is_ai_turn = False
        self.winning_line_coords = None  # Store start and end positions for the winning line
        self.message_text = 'Press or say a grid location 1-9\n(from top left to bottom right) to make a move.'
        self.message = tk.Label(self.root, text=self.message_text, font=("Arial", 12, "bold"), fg="white", bg="#2e2e2e")
        self.currentLanguage = language
        self.message.pack(pady=20)  # Place it once in the desired position
        # self.canvas.bind("<Button-1>", self.handle_click)
        root.bind("<KeyPress>", self.handle_key_input)
    
    def speech_to_text(self, lang):        
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
    
    def convert_to_int(self, userInput):

        userInput = userInput.strip().lower()

        numbers = {
            'one': 1, 
            'two': 2, 
            'three': 3, 
            'four': 4, 
            'five': 5, 
            'six': 6, 
            'seven': 7, 
            'eight': 8, 
            'nine': 9,
        }

        if userInput in numbers.keys():
            return numbers(userInput)
        
        else:
            return None
    
    def handle_speech_input(self):
        if self.winner or self.game_steps == 9:
            return  # Stop if game is over

        if not self.is_ai_turn:
            try:
                userInput = self.speech_to_text(self.currentLanguage)
                userInput = self.convert_to_int(userInput)
                if (userInput):
                    move = int(userInput)
                    if 1 <= move <= 9:
                        row, col = self.pos_to_cell(move)
                        if self.game_array[row][col] == INF:
                            self.game_array[row][col] = 1  # Player move
                            self.check_winner()
                            self.is_ai_turn = True
                            self.game_steps += 1
            except ValueError:
                pass  # Ignore non-numeric input

    def handle_key_input(self, event):
        if self.winner or self.game_steps == 9:
            return  # Stop if game is over

        if not self.is_ai_turn:
            try:
                move = int(event.char)
                if 1 <= move <= 9:
                    row, col = self.pos_to_cell(move)
                    if self.game_array[row][col] == INF:
                        self.game_array[row][col] = 1  # Player move
                        self.check_winner()
                        self.is_ai_turn = True
                        self.game_steps += 1
            except ValueError:
                pass  # Ignore non-numeric input
    
    def pos_to_cell(self, move):
        return (move - 1) // 3, (move - 1) % 3

    def get_ai_move(self):
        board_state = ""
        for row in self.game_array:
            for cell in row:
                board_state += "-" if cell == INF else ("X" if cell == 1 else "O")
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are playing Tic Tac Toe. Provide move as a single digit 1-9."},
                    {"role": "user", "content": f"Current board: {board_state}. What is your move?"}
                ]
            )
            move = int(completion.choices[0].message.content.strip())
            return (move - 1) // 3, (move - 1) % 3
        except:
            for i in range(3):
                for j in range(3):
                    if self.game_array[i][j] == INF:
                        return i, j

    def check_winner(self):
        lines = [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
        ]
        for line in lines:
            values = [self.game_array[i][j] for i, j in line]
            if values == [1, 1, 1]:
                self.winner = 'Player'
                self.winning_line_coords = line  # Store the winning line
            elif values == [0, 0, 0]:
                self.winner = 'Chatbot'
                self.winning_line_coords = line  # Store the winning line

    def draw_x(self, row, col):
        x1, y1 = col * CELL_SIZE, row * CELL_SIZE
        x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
        self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=4)
        self.canvas.create_line(x1, y2, x2, y1, fill="blue", width=4)

    def draw_o(self, row, col):
        x1, y1 = col * CELL_SIZE + 5, row * CELL_SIZE + 5
        x2, y2 = x1 + CELL_SIZE - 10, y1 + CELL_SIZE - 10
        self.canvas.create_oval(x1, y1, x2, y2, outline="green", width=4)
    
    def draw_winning_line(self):
        if self.winning_line_coords:
            start = self.winning_line_coords[0]
            end = self.winning_line_coords[-1]
            
            x1 = start[1] * CELL_SIZE + CELL_SIZE // 2
            y1 = start[0] * CELL_SIZE + CELL_SIZE // 2
            x2 = end[1] * CELL_SIZE + CELL_SIZE // 2
            y2 = end[0] * CELL_SIZE + CELL_SIZE // 2

            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=5)
    
    def draw(self):
        self.canvas.delete("all")
        for i in range(1, 3):
            self.canvas.create_line(0, i * CELL_SIZE, CANVAS_SIZE, i * CELL_SIZE, fill="black")
            self.canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, CANVAS_SIZE, fill="black")
        for row in range(3):
            for col in range(3):
                if self.game_array[row][col] == 1:
                    self.draw_x(row, col)
                elif self.game_array[row][col] == 0:
                    self.draw_o(row, col)
        if self.winner:
            self.draw_winning_line()
            self.message_text = self.winner + " wins!"
        self.message.config(text=self.message_text)  # Update message content

    def run_game_process(self):
        if self.winner:
            return
        if self.game_steps == 9:
            self.message_text = "Tie!"
            return
        if self.is_ai_turn:
            row, col = self.get_ai_move()
            if self.game_array[row][col] == INF:
                self.game_array[row][col] = 0
                self.game_steps += 1
                self.check_winner()
                self.is_ai_turn = False

    def run(self):
        self.draw()
        self.run_game_process()
        # self.handle_speech_input()
        self.canvas.after(100, self.run)
