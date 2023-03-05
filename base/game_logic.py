from base.start import *

# Looking up a result of moving one row left in the pre-calculated dictionary is
# about much faster than calculating it every time.
# If we optimistically take 32768 tile as a maximum of what we expect to encounter on the board,
# the table has the same 2**16 = 65536 entries. We need 4 such tables. Not much memory for a nice speedup.


def random_eval(row, score):
    return np.random.random()


def score_eval(row, score):
    return score


def get_max_tile(row):
    return int(1 << np.max(row))


def create_table():
    table = {}
    for a in range(16):
        for b in range(16):
            for c in range(16):
                for d in range(16):
                    score = 0
                    line = (a, b, c, d)
                    if (len(set(line)) == 4 and min(line)) or (not max(line)):
                        table[line] = (line, score, False)
                        continue
                    line_1 = [v for v in line if v]
                    for i in range(len(line_1) - 1):
                        x = line_1[i]
                        if x == line_1[i + 1]:
                            score += 1 << (x + 1)
                            line_1[i], line_1[i + 1] = x + 1, 0
                    line_2 = [v for v in line_1 if v]
                    line_2 = line_2 + [0] * (4 - len(line_2))
                    table[line] = (line_2, score, line != tuple(line_2))
    return table


TABLE = create_table()

#   member of the class is the state of the 4*4 board,
#   score = current score in the game
#   odometer = number of moves from the start
#   row = numpy array of shape (4, 4)
#   numbers stored in the Game are 0 for 0 and log2(n) for 2,4,8 ..


class Game:

    debug_actions = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}

    def __init__(self, params=None):
        self.idx = None
        self.player = None
        self.row = None
        self.initial = None
        self.score = 0
        self.odo = 0
        self.moves = []
        self.tiles = []
        self.generate(params)

    def __str__(self):
        return '\n'.join([''.join([str(1 << val if val else 0) + '\t' * (4 if (1 << val) < 1000 else 3)
                                   for val in j]) for j in self.row]) \
               + f'\n score = {str(self.score)}, moves = {str(self.odo)}, reached {get_max_tile(self.row)}'

    @staticmethod
    def empty(row):
        return np.where(row == 0)

    def new_tile(self):
        i, j = self.empty(self.row)
        tile = 1 if random.randrange(10) else 2
        pos = random.randrange(len(i))
        self.row[i[pos], j[pos]] = tile
        self.tiles.append([i[pos], j[pos], tile])

    def generate(self, params: dict):
        if params is None:
            self.row = np.zeros((4, 4), dtype=np.int32)
            self.new_tile()
            self.new_tile()
            self.tiles = []
            self.initial = self.row.tolist()
        else:
            self.idx = params.get('idx', None)
            self.player = params.get('player', None)
            self.initial = params.get('initial', None)
            self.row = np.array(params['row'], dtype=np.int32)
            self.score = params['score']
            self.odo = params['odo']
            self.moves = params.get('moves', [])
            self.tiles = params.get('tiles', [])

    @staticmethod
    def tiles_to_int(tiles: list):
        return [[int(v[0]), int(v[1]), v[2]] for v in tiles]

    def to_dict(self):
        return {
            'idx': self.idx,
            'player': self.player,
            'initial': self.initial,
            'current': self.row.tolist(),
            'score': self.score,
            'num_of_moves': self.odo,
            'max_tile': get_max_tile(self.row),
            'moves': self.moves,
            'tiles':  self.tiles_to_int(self.tiles)
        }

    @staticmethod
    def empty_count(row):
        return 16 - np.count_nonzero(row)

    @staticmethod
    def adjacent_pair_count(row):
        return 24 - np.count_nonzero(row[:, :3] - row[:, 1:]) - np.count_nonzero(row[:3, :] - row[1:, :])

    def game_over(self, row):
        if self.empty_count(self.row):
            return False
        return not self.adjacent_pair_count(row)

    @staticmethod
    def _left(row, score):
        change = False
        new_row = row.copy()
        new_score = score
        for i in range(4):
            line, score, change_line = TABLE[tuple(row[i])]
            if change_line:
                change = True
                new_score += score
                new_row[i] = line
        return new_row, new_score, change

    def pre_move(self, row, score, direction):
        new_row = np.rot90(row, direction) if direction else row
        new_row, new_score, change = self._left(new_row, score)
        if direction:
            new_row = np.rot90(new_row, 4 - direction)
        return new_row, new_score, change

    def _move_on(self, best_dir, best_row, best_score):
        self.moves.append(best_dir)
        self.odo += 1
        self.row, self.score = best_row, best_score
        self.new_tile()

    # looking a few moves ahead and branching several new tile positions randomly
    def look_forward(self, estimator, row, score, depth, width, trigger):
        if depth == 0:
            return estimator(row, score)
        empty = self.empty_count(row)
        if empty >= trigger:
            return estimator(row, score)
        num_tiles = min(width, empty)
        empty_i, empty_j = self.empty(row)
        tile_positions = random.sample(range(len(empty_i)), num_tiles)
        average = 0
        for pos in tile_positions:
            new_tile = 1 if random.randrange(10) else 2
            new_row = row.copy()
            new_row[empty_i[pos], empty_j[pos]] = new_tile
            if self.game_over(new_row):
                best_value = 0
            else:
                best_value = - np.inf
                for direction in range(4):
                    test_row, test_score, change = self.pre_move(new_row, score, direction)
                    if change:
                        value = self.look_forward(estimator, test_row, test_score,
                                                  depth=depth - 1, width=width, trigger=trigger)
                        best_value = max(best_value, value)
            average += max(best_value, 0)
        average = average / num_tiles
        return average

    def find_best_move(self, estimator, depth, width, trigger):
        best_dir, best_value = 0, - np.inf
        best_row, best_score = None, None
        for direction in range(4):
            new_row, new_score, change = self.pre_move(self.row, self.score, direction)
            if change:
                value = self.look_forward(estimator, new_row, new_score,
                                          depth=depth, width=width, trigger=trigger)
                if value > best_value:
                    best_dir, best_value = direction, value
                    best_row, best_score = new_row, new_score
        return best_dir, best_row, best_score

    # Run single episode (for debugging purposes)
    def trial_run_debug(self, estimator, depth=0, width=1, trigger=0):
        print('Starting position:')
        print(self)
        while True:
            if self.game_over(self.row):
                return
            best_dir, best_row, best_score = self.find_best_move(estimator, depth, width, trigger)
            self._move_on(best_dir, best_row, best_score)
            print(f'On {self.odo} we moved {self.debug_actions[best_dir]}')
            print(self)

    def trial_run(self, estimator, depth=0, width=1, trigger=0):
        while True:
            if self.game_over(self.row):
                return
            best_dir, best_row, best_score = self.find_best_move(estimator, depth, width, trigger)
            self._move_on(best_dir, best_row, best_score)

    def watch_run(self, job_idx, estimator, depth=0, width=1, trigger=0):
        check = time.time() + 2
        last_move = 0
        BACK.update_watch_job(job_idx, [], [])
        while True:
            if self.game_over(self.row):
                print(self)
                self.moves.append(-1)
                BACK.update_watch_job(job_idx,
                                      moves=self.moves[last_move:], tiles=self.tiles_to_int(self.tiles[last_move:]))
                return
            best_dir, best_row, best_score = self.find_best_move(estimator, depth, width, trigger)
            self._move_on(best_dir, best_row, best_score)
            if time.time() > check:
                if BACK.new_watch_job(job_idx):
                    return
                BACK.update_watch_job(job_idx,
                                      moves=self.moves[last_move:], tiles=self.tiles_to_int(self.tiles[last_move:]))
                last_move = self.odo
                check = time.time() + 2


def replay_debug(game_idx: str, end=1000000):
    game_dict = BACK.get_item(game_idx, 'Games')
    game_dict['row'] = game_dict['initial']
    print(f"total moves = {game_dict['num_of_moves']}, score = {game_dict['score']}")
    game_dict['score'] = 0
    game_dict['odo'] = 0
    game = Game(game_dict)
    c, move = 0, 0
    print(game)
    while True:
        move = game.moves[c]
        if move == -1:
            break
        i, j, tile = game.tiles[c]
        print(move, i, j, tile)
        new_row, new_score, change = game.pre_move(game.row, game.score, move)
        if not change:
            print(f'NO MOVE! {c}, {move}')
            break
        game.row = new_row
        if game.row[i, j]:
            print(f'TILE SPACE OCCUPIED! {c}, {i}, {j}, {tile}')
            break
        game.row[i, j] = tile
        game.score = new_score
        game.odo += 1
        print(game)
        c += 1
        if c == end:
            break
