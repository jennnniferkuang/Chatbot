import pygame as pg
import sys
from random import randint
from openai import OpenAI
from pathlib import Path

WIN_SIZE = 600
CELL_SIZE = WIN_SIZE // 3
INF = float('inf')
vec2 = pg.math.Vector2
CELL_CENTER = vec2(CELL_SIZE / 2)


class TicTacToe:
    def __init__(self, game):
        self.game = game
        self.field_image = self.get_scaled_image(path='resources/field.png', res=[WIN_SIZE] * 2)
        self.O_image = self.get_scaled_image(path='resources/o.png', res=[CELL_SIZE] * 2)
        self.X_image = self.get_scaled_image(path='resources/x.png', res=[CELL_SIZE] * 2)

        self.game_array = [[INF, INF, INF],
                           [INF, INF, INF],
                           [INF, INF, INF]]
        self.player = 1

        self.line_indices_array = [[(0, 0), (0, 1), (0, 2)],
                                   [(1, 0), (1, 1), (1, 2)],
                                   [(2, 0), (2, 1), (2, 2)],
                                   [(0, 0), (1, 0), (2, 0)],
                                   [(0, 1), (1, 1), (2, 1)],
                                   [(0, 2), (1, 2), (2, 2)],
                                   [(0, 0), (1, 1), (2, 2)],
                                   [(0, 2), (1, 1), (2, 0)]]
        self.winner = None
        self.game_steps = 0
        self.font = pg.font.SysFont('Verdana', CELL_SIZE // 4, True)
        self.input_text = ""
        self.input_active = False  # Track whether the textbox is active
        self.input_box = pg.Rect(10, WIN_SIZE - 60, 200, 50)
        self.input_color = pg.Color('lightskyblue3')
        self.txt_surface = self.font.render(self.input_text, True, self.input_color)
        self.client = OpenAI(api_key="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")
        self.is_ai_turn = False

    def get_ai_move(self):
        print("ChatGPT's turn...")
        # Convert game state to string representation
        board_state = ""
        for row in self.game_array:
            for cell in row:
                if cell == INF:
                    board_state += "-"
                elif cell == 1:
                    board_state += "X"
                else:
                    board_state += "O"

        # Ask ChatGPT for the next move
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are playing TicTacToe. The board is numbered 1-9 from left to right, top to bottom. Only respond with a single number representing your move. If you can't come up with a answer in 5 seconds, just choose a random move that is available."},
                {"role": "user", "content": f"Current board: {board_state}. What's your move (1-9)?"}
            ]
        )
        
        try:
            move = int(completion.choices[0].message.content)
            row = (move - 1) // 3
            col = (move - 1) % 3
            return row, col
        except:
            # Fallback to first available move if AI response is invalid
            for i in range(3):
                for j in range(3):
                    if self.game_array[i][j] == INF:
                        return i, j

    def handle_text_input(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # Check if the click is inside the input box
            if self.input_box.collidepoint(event.pos):
                self.input_active = True  # Activate the textbox
                self.input_color = pg.Color('dodgerblue2')
            else:
                self.input_active = False  # Deactivate the textbox
                self.input_color = pg.Color('lightskyblue3')

        if event.type == pg.KEYDOWN and self.input_active:
            if event.key == pg.K_RETURN:
                self.input_active = False  # Deactivate the textbox after submitting
                try:
                    pos = int(self.input_text)
                    if 1 <= pos <= 9:
                        row = (pos - 1) // 3
                        col = (pos - 1) % 3
                        if self.game_array[row][col] == INF and not self.winner:
                            self.game_array[row][col] = self.player
                            self.player = not self.player
                            self.game_steps += 1
                            self.check_winner()
                            self.is_ai_turn = True
                except ValueError:
                    pass
                self.input_text = ''
            elif event.key == pg.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                if len(self.input_text) < 1:
                    self.input_text += event.unicode
            self.txt_surface = self.font.render(self.input_text, True, self.input_color)

    def handle_direct_input(self, input_text):
        try:
            pos = int(input_text)
            if 1 <= pos <= 9:
                row = (pos - 1) // 3
                col = (pos - 1) % 3
                if self.game_array[row][col] == INF and not self.winner:
                    self.game_array[row][col] = self.player
                    self.player = not self.player
                    self.game_steps += 1
                    self.check_winner()
                    self.is_ai_turn = True
        except ValueError:
            pass
            
    def check_winner(self):
        for line_indices in self.line_indices_array:
            sum_line = sum([self.game_array[i][j] for i, j in line_indices])
            if sum_line in {0, 3}:
                self.winner = 'XO'[sum_line == 0]
                self.winner_line = [vec2(line_indices[0][::-1]) * CELL_SIZE + CELL_CENTER,
                                    vec2(line_indices[2][::-1]) * CELL_SIZE + CELL_CENTER]

    def run_game_process(self):
        if self.winner or self.game_steps == 9:
            return

        if self.is_ai_turn:
            row, col = self.get_ai_move()
            if self.game_array[row][col] == INF:
                self.game_array[row][col] = self.player
                self.player = not self.player
                self.game_steps += 1
                self.check_winner()
                self.is_ai_turn = False
        elif not self.input_active:
            current_cell = vec2(pg.mouse.get_pos()) // CELL_SIZE
            col, row = map(int, current_cell)
            left_click = pg.mouse.get_pressed()[0]

            if left_click and self.game_array[row][col] == INF:
                self.game_array[row][col] = self.player
                self.player = not self.player
                self.game_steps += 1
                self.check_winner()
                self.is_ai_turn = True

    def draw(self):
        self.game.screen.blit(self.field_image, (0, 0))
        self.draw_objects()
        self.draw_winner()
        # Draw input box
        pg.draw.rect(self.game.screen, self.input_color, self.input_box, 2)
        self.game.screen.blit(self.txt_surface, (self.input_box.x + 5, self.input_box.y + 5))
        hint_text = self.font.render("Enter 1-9:", True, 'white')
        self.game.screen.blit(hint_text, (10, WIN_SIZE - 90))
        
    def draw_objects(self):
        for y, row in enumerate(self.game_array):
            for x, obj in enumerate(row):
                if obj != INF:
                    self.game.screen.blit(self.X_image if obj else self.O_image, vec2(x, y) * CELL_SIZE)

    def draw_winner(self):
        if self.winner:
            pg.draw.line(self.game.screen, 'red', *self.winner_line, CELL_SIZE // 8)
            label = self.font.render(f'Player "{self.winner}" wins!', True, 'white', 'black')
            self.game.screen.blit(label, (WIN_SIZE // 2 - label.get_width() // 2, WIN_SIZE // 4))

    def draw(self):
        self.game.screen.blit(self.field_image, (0, 0))
        self.draw_objects()
        self.draw_winner()
        # Draw input box
        pg.draw.rect(self.game.screen, self.input_color, self.input_box, 2)
        self.game.screen.blit(self.txt_surface, (self.input_box.x + 5, self.input_box.y + 5))
        hint_text = self.font.render("Type 1-9:", True, 'white')
        self.game.screen.blit(hint_text, (10, WIN_SIZE - 90))

    @staticmethod
    def get_scaled_image(path, res):
        img = pg.image.load(path)
        return pg.transform.smoothscale(img, res)

    def print_caption(self):
        pg.display.set_caption(f'Player "{"OX"[self.player]}" turn!')
        if self.winner:
            pg.display.set_caption(f'Player "{self.winner}" wins! Press Space to Restart')
        elif self.game_steps == 9:
            pg.display.set_caption(f'Game Over! Press Space to Restart')

    def run(self):
        self.print_caption()
        self.draw()
        self.run_game_process()


class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode([WIN_SIZE] * 2)
        self.clock = pg.time.Clock()
        self.tic_tac_toe = TicTacToe(self)

    def new_game(self):
        self.tic_tac_toe = TicTacToe(self)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    self.new_game()
            self.tic_tac_toe.handle_text_input(event)

    def run(self):
        while True:
            self.tic_tac_toe.run()
            self.check_events()
            user_input = self.get_user_input()  # Method to get user input (e.g., from speech-to-text)
            if user_input:
                self.tic_tac_toe.handle_direct_input(user_input)
            pg.display.update()
            self.clock.tick(60)

    def get_user_input(self):
        # Placeholder for the method to get user input (e.g., from speech-to-text)
        # Return the user input as a string (e.g., "9")
        return "9"

if __name__ == '__main__':
    game = Game()
    game.run()