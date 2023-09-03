from .r_learning import *


class QAgentRun:

    def __init__(self, name: str, no_logs=True):
        self.name = name
        self.weights = None
        self.features = None

        self.print = no_log_function if no_logs else print
        match self.name:
            case 'Random Moves':
                self.evaluate = random_eval
            case 'Best Score':
                self.evaluate = score_eval
            case _:
                agent = BACK.get_agent(self.name)
                if not agent['weightSignature']:
                    self.evaluate = random_eval
                else:
                    n = agent['N']
                    self.features = FEATURE_FUNCTIONS[n]
                    self.load_weights()
                    self.evaluate = self._evaluate

    def load_weights(self):
        self.print('loading weights ...')
        w = BACK.s3_load(self.name)
        if w:
            self.weights = []
            for weight_component in w:
                self.weights += weight_component.tolist()
            del w

    def _evaluate(self, row, score=None):
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])


def watch_run(job: dict):
    user = job['user']
    agent = job['name']
    depth = job['depth']
    width = job['width']
    trigger = job['trigger']
    estimator = QAgentRun(name=agent).evaluate
    BACK.launch_watch_job(user)
    game_params = {**job['startGame'], 'user': user}
    game = Game(game_params)
    BACK.games.replace_one({'user': user}, game.to_mongo(), upsert=True)
    check = time.time() + SAVE_NEW_MOVES
    last_move = 0
    initial_moves = game.numMoves
    while not game.game_over(game.row):
        best_dir, best_row, best_score = game.find_best_move(estimator, depth, width, trigger)
        game._move_on(best_dir, best_row, best_score)
        if time.time() > check:
            BACK.update_watch_game(user, game.moves[last_move:], game.tiles_to_int(game.tiles[last_move:]))
            last_move = game.numMoves - initial_moves
            check = time.time() + 2
    game.moves.append(-1)
    BACK.update_watch_game(user, game.moves[last_move:], game.tiles_to_int(game.tiles[last_move:]))
    time.sleep(10)
    BACK.games.delete_one({'user': user})
