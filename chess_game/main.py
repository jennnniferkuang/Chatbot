import tkinter as tk
from tkinter import messagebox
import chess
import openai

# dimensions, change to adjust this is fine for my computer but idk the design
# also very important, image scaling is flat so just resize the images themselves, easier this way
WIDTH, HEIGHT = 600, 600
CELL_SIZE = WIDTH // 8

# colour scheme copied from online lol
WHITE = "#FFFFFF"
BLACK = "#000000"
GREEN = "#AAFF00"
LIGHT_BROWN = "#F0D9B5"
DARK_BROWN = "#B58863"

# all images of the pieces in resources, just reference that 
# k is king n is knight everything else is first letter of piece (b = black w = white second character varies based on piece name)
PIECES = {
    'P': 'chess_game/resources/wp.png',
    'p': 'chess_game/resources/bp.png',
    'R': 'chess_game/resources/wr.png',
    'r': 'chess_game/resources/br.png',
    'N': 'chess_game/resources/wn.png',
    'n': 'chess_game/resources/bn.png',
    'B': 'chess_game/resources/wb.png',
    'b': 'chess_game/resources/bb.png',
    'Q': 'chess_game/resources/wq.png',
    'q': 'chess_game/resources/bq.png',
    'K': 'chess_game/resources/wk.png',
    'k': 'chess_game/resources/bk.png'
}

board = chess.Board()

class ChessGame:
    def __init__(self, root):
        self.root = root
        # change
        self.root.title("chess")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT + 50)
        self.canvas.pack()
        self.input_text = tk.StringVar()
        self.valid_move = True
        
        # failsafe
        self.player_side = "white"
        self.ai_side = "black"
        self.is_ai_turn = False

        self.piece_images = {}
        for key, path in PIECES.items():
            self.piece_images[key] = tk.PhotoImage(file=path).subsample(8, 8)

        self.draw_board()
        self.update_pieces()

        # input box, the green thing, delete this later
        self.input_box = tk.Entry(root, textvariable=self.input_text, font=("Arial", 24), bg=GREEN, fg=BLACK)
        self.canvas.create_window(WIDTH // 2, HEIGHT + 25, window=self.input_box, width=WIDTH - 20, height=40)
        self.input_box.bind("<Return>", self.process_move)

        # you can now choose the side
        self.side_choice = tk.StringVar(value="Choose Side")
        self.side_menu = tk.OptionMenu(root, self.side_choice, "white", "black", command=self.set_player_side)
        self.side_menu.pack()
        openai.api_key = "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" 

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
                x1, y1 = col * CELL_SIZE, row * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def update_pieces(self):
        self.canvas.delete("piece")
        for row in range(8):
            for col in range(8):
                piece = board.piece_at(row * 8 + col)
                if piece:
                    piece_symbol = piece.symbol()
                    x = col * CELL_SIZE + CELL_SIZE // 2
                    y = row * CELL_SIZE + CELL_SIZE // 2
                    self.canvas.create_image(x, y, image=self.piece_images[piece_symbol], tags="piece")

        self.check_game_state()

    def check_game_state(self):
        if board.is_checkmate():
            winner = "AI" if board.turn == (self.player_side == "white") else "Player"
            messagebox.showinfo("Game Over", f"Checkmate! {winner} wins.")
        elif board.is_stalemate():
            messagebox.showinfo("Game Over", "Stalemate!")
        elif board.is_check():
            messagebox.showinfo("Alert", "In Check!")

    def process_move(self, event=None):
        global board
        move = self.input_text.get()
        try:
            board.push_san(move)
            self.valid_move = True
            self.update_pieces()
            self.input_text.set("")

            # self explanatory
            if board.turn == (self.ai_side == "white"):
                self.ai_make_move()
        except ValueError:
            self.valid_move = False
            messagebox.showerror("Error", "Illegal move!")
            self.input_text.set("")

    def set_player_side(self, side):
        self.player_side = side
        self.ai_side = "white" if side == "black" else "black"
        self.is_ai_turn = self.ai_side == "white"
        # to prevent flip flopping and other garbage that might happen
        self.side_menu.config(state="disabled")

        # if ai is white
        if self.is_ai_turn:
            self.ai_make_move()

    def ai_make_move(self):
        global board
        try:
            prompt = f"The current board position in FEN notation is: {board.fen()}. You are playing as {self.ai_side}. Provide the best move in standard chess notation where the notation is only the piece and where the piece moves to (example, pawn to e2 = e2, knight to f3 = Nf3, queen to a5 = Qa5). Be decent at the game, but not an completely optimized player. Play like a 1600 elo player."
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a chess bot. Respond with a move in standard chess notation. Be decent at the game, but not an completely optimized player. Play like a 1600 elo player."},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_move = response['choices'][0]['message']['content'].strip()
            board.push_san(ai_move)
            self.update_pieces()
        except Exception as e:
            # should never happen, its not like depth 24 stockfish
            messagebox.showerror("Error", f"AI move failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    game = ChessGame(root)
    root.mainloop()
