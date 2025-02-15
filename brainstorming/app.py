# basic chess, images not included but if you upload them it works, also for movement its 
# first letter of piece name (capitalized, knight is N, king is K, pawn is nothing)
# second letter and third number is where the piece moves to
import pygame as pg
import sys
import chess

pg.init()

# dimensions, change to adjust this is fine for my computer but idk the design
WIDTH, HEIGHT = 800, 800
CELL_SIZE = WIDTH // 8
# font can be adjusted
FONT = pg.font.SysFont('Arial', 24)

# colour scheme copied from online lol
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)

# all images of the pieces in resources, just reference that 
# k is king n is knight everything else is first letter of piece (b = black w = white second character varies based on piece name)
PIECES = {
    'P': pg.image.load('resources/wp.png'),
    'p': pg.image.load('resources/bp.png'),
    'R': pg.image.load('resources/wr.png'),
    'r': pg.image.load('resources/br.png'),
    'N': pg.image.load('resources/wn.png'),
    'n': pg.image.load('resources/bn.png'),
    'B': pg.image.load('resources/wb.png'),
    'b': pg.image.load('resources/bb.png'),
    'Q': pg.image.load('resources/wq.png'),
    'q': pg.image.load('resources/bq.png'),
    'K': pg.image.load('resources/wk.png'),
    'k': pg.image.load('resources/bk.png')
}

# again idk the size
for key in PIECES:
    PIECES[key] = pg.transform.scale(PIECES[key], (CELL_SIZE, CELL_SIZE))

board = chess.Board()

class ChessGame:
    def __init__(self):
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption("Chess Game")
        self.input_text = ""
        self.input_box = pg.Rect(10, HEIGHT - 40, 780, 30)
        self.input_color = LIGHT_BROWN
        self.valid_move = True

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
                pg.draw.rect(self.screen, color, pg.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = board.piece_at(row * 8 + col)
                if piece:
                    piece_symbol = piece.symbol()
                    self.screen.blit(PIECES[piece_symbol], (col * CELL_SIZE, row * CELL_SIZE))

    # delete this later, this is just the box used for inputs
    def draw_input_box(self):
        pg.draw.rect(self.screen, self.input_color, self.input_box, 2)
        txt_surface = FONT.render(self.input_text, True, WHITE)
        self.screen.blit(txt_surface, (self.input_box.x + 10, self.input_box.y + 5))

        # replace this with chatgpt input for invalid moves
        if not self.valid_move:
            invalid_txt = FONT.render("illegal move", True, (255, 0, 0))
            self.screen.blit(invalid_txt, (10, HEIGHT - 70))
            
        # add something to display the actual loss/win, this just calculates it
        if board.is_checkmate():
          state_txt = FONT.render("checkmate", True, (255, 255, 0))
          self.screen.blit(state_txt, (10, HEIGHT - 100))
        elif board.is_stalemate():
          state_txt = FONT.render("stalemate", True, (255, 255, 0))
          self.screen.blit(state_txt, (10, HEIGHT - 100))
        elif board.is_check():
          state_txt = FONT.render("in check", True, (255, 255, 0))
          self.screen.blit(state_txt, (10, HEIGHT - 100))
    # again, change because of ai also the first piece name is capitalized to make it consistent with chess notation
    def handle_input(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.input_box.collidepoint(event.pos):
                self.input_color = DARK_BROWN
            else:
                self.input_color = LIGHT_BROWN

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                self.process_move()
                self.input_text = ""
            elif event.key == pg.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                self.input_text += event.unicode

    def process_move(self):
        global board
        try:
            board.push_san(self.input_text)
            self.valid_move = True
        except ValueError:
            self.valid_move = False

    def run(self):
        # running the game itself
        clock = pg.time.Clock()
        while True:
            self.screen.fill(BLACK)
            self.draw_board()
            self.draw_pieces()
            self.draw_input_box()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                self.handle_input(event)

            pg.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    game = ChessGame()
    game.run()
