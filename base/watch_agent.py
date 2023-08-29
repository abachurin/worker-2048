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


def run_game(game_params: dict, estimator: Callable[[np.ndarray, int], float],
             depth: int, width: int, trigger: int):
    game = Game(game_params)
    name = game.name
    BACK.save_game(name, game.to_mongo())
    check = time.time() + SAVE_NEW_MOVES
    last_move = 0
    initial_moves = game.numMoves
    block_new = True
    while True:
        best_dir, best_row, best_score = game.find_best_move(estimator, depth, width, trigger)
        game._move_on(best_dir, best_row, best_score)
        is_over = game.game_over(game.row)
        if is_over:
            game.moves.append(-1)
        if is_over or time.time() > check:
            if STOPPER[name]:
                return
            BACK.update_watch_game(name, game.moves[last_move:], game.tiles_to_int(game.tiles[last_move:]))
            last_move = game.numMoves - initial_moves
            if block_new:
                BACK.set_watch_job(game.user, False)
                block_new = False
            check = time.time() + 2
        if is_over:
            return


def watch_run(job: dict):
    user = job['user']
    agent = job['name']
    depth = job['depth']
    width = job['width']
    trigger = job['trigger']
    estimator = QAgentRun(name=agent).evaluate
    BACK.launch_watch_job(user)
    BACK.admin_adjust_memo(user)
    time_to_sleep = time.time() + WAIT_TO_CONTINUE_WATCH
    current_game = ''
    current = None
    while True:
        new_game_params = BACK.get_watch_game(user)
        match new_game_params['status']:
            case NewGameJob.KILL:
                break
            case NewGameJob.RESTART:
                if current_game:
                    STOPPER[current_game] = True
                    BACK.games.delete_one({'name': current_game})
                current_game = new_game_params['name']
                BACK.set_watch_job(user, False)
                STOPPER[new_game_params['name']] = False
                current = Thread(target=run_game, args=(new_game_params, estimator, depth, width, trigger))
                current.start()
            case _:
                if current and not current.is_alive():
                    time_to_sleep = time.time() + WAIT_TO_CONTINUE_WATCH
                    current = None
                if time.time() > time_to_sleep - 10:
                    BACK.set_watch_job(user, True)
                if time.time() > time_to_sleep:
                    break
        time.sleep(2)
    BACK.games.delete_many({'user': user})
    return

