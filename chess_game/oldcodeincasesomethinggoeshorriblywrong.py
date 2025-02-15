import tkinter as tk
from tkinter import messagebox
import chess

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

        self.piece_images = {}
        for key, path in PIECES.items():
            self.piece_images[key] = tk.PhotoImage(file=path).subsample(8, 8)

        self.draw_board()
        self.update_pieces()

        # input box, the green thing, delete this later
        self.input_box = tk.Entry(root, textvariable=self.input_text, font=("Arial", 24), bg=GREEN, fg=BLACK)
        self.canvas.create_window(WIDTH // 2, HEIGHT + 25, window=self.input_box, width=WIDTH - 20, height=40)
        self.input_box.bind("<Return>", self.process_move)

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

    # add something to display the actual loss/win, this just calculates it
    def check_game_state(self):
        if board.is_checkmate():
            messagebox.showinfo("Game Over", "Checkmate!")
        elif board.is_stalemate():
            messagebox.showinfo("Game Over", "Stalemate!")
        elif board.is_check():
            messagebox.showinfo("Alert", "In Check!")

    # again, change because of ai also the first piece name is capitalized to make it consistent with chess notation
    def process_move(self, event=None):
        global board
        move = self.input_text.get()
        try:
            board.push_san(move)
            self.valid_move = True
            self.update_pieces()
        except ValueError:
            self.valid_move = False
            messagebox.showerror("Error", "Illegal move!")
        self.input_text.set("")


if __name__ == "__main__":
    root = tk.Tk()
    game = ChessGame(root)
    root.mainloop()
