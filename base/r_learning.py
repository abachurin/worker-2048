from .game_logic import *


# Argument x for feature functions is a (4, 4) numpy array of np.int32
# features = all adjacent pairs

def f_2(x: np.ndarray) -> np.ndarray:
    x_vert = ((x[:3, :] << 4) + x[1:, :]).ravel()
    x_hor = ((x[:, :3] << 4) + x[:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor])


# features = all adjacent triples, i.e. 3 in a row + 3 in any square missing one corner
def f_3(x: np.ndarray) -> np.ndarray:
    x_vert = ((x[:2, :] << 8) + (x[1:3, :] << 4) + x[2:, :]).ravel()
    x_hor = ((x[:, :2] << 8) + (x[:, 1:3] << 4) + x[:, 2:]).ravel()
    x_ex_00 = ((x[1:, :3] << 8) + (x[1:, 1:] << 4) + x[:3, 1:]).ravel()
    x_ex_01 = ((x[:3, :3] << 8) + (x[1:, :3] << 4) + x[1:, 1:]).ravel()
    x_ex_10 = ((x[:3, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_ex_11 = ((x[:3, :3] << 8) + (x[1:, :3] << 4) + x[:3, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_ex_00, x_ex_01, x_ex_10, x_ex_11])


# Initially I also made all adjacent quartets of different shape, but the learning was not happening.
# My theory is that: 1) we want our features to intersect and correlate (otherwise we will only learn
# several pieces of the board, and that obviously can not lead to anything.
# but 2) we don't want them to intersect too much (like 3 cells common to two quartets), as they start
# to kinda suppress and contradict each other.
# So I left just columns, rows and squares. 17 features all in all. And it works just fine.
def f_4(x: np.ndarray) -> np.ndarray:
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq])


# Finally, we try adding 4 "cross" 5-features for middle cells
def f_5(x: np.ndarray) -> np.ndarray:
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1: 3, 1: 3] << 16) + (x[:2, 1: 3] << 12) + (x[1: 3, :2] << 8) + (x[2:, 1: 3] << 4) + x[1: 3, 2:]
                ).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle])


# Adding some limited 6-features, up to < 2 ** (cutoff - 1) > tile
def f_6(x: np.ndarray) -> np.ndarray:
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1: 3, 1: 3] << 16) + (x[:2, 1: 3] << 12) + (x[1: 3, :2] << 8) + (x[2:, 1: 3] << 4) + x[1: 3, 2:]
                ).ravel()
    y = np.minimum(x, 13)
    x_vert_6 = (537824 * y[0: 2, 0: 3] + 38416 * y[1: 3, 0: 3] + 2744 * y[2:, 0: 3] + 196 * y[0: 2, 1:] +
                14 * y[1: 3, 1:] + y[2:, 1:]).ravel()
    x_hor_6 = (537824 * y[0: 3, 0: 2] + 38416 * y[0: 3, 1: 3] + 2744 * y[0: 3, 2:] + 196 * y[1:, 0: 2] +
               14 * y[1:, 1: 3] + y[1:, 2:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle, x_vert_6, x_hor_6])


FEATURE_FUNCTIONS = {
    2: f_2,
    3: f_3,
    4: f_4,
    5: f_5,
    6: f_6
}
PAR_SHAPE = {
    2: (24, 16 ** 2),
    3: (52, 16 ** 3),
    4: (17, 16 ** 4),
    5: (21, 16 ** 5),
    6: (33, 0)
}
CUTOFF_FOR_6_F = 14
EXTRA_AGENTS = ['Random Moves', 'Best Score']

# The RL agent. It is not actually Q, as it tries to learn values of the states (V), rather than actions (Q).
# Not sure what is the correct terminology here, this is definitely a TD(0), basically a modified Q-learning.
# The important details:
# 1) The discount parameter gamma = 1. Don't see why discount rewards in this episodic task.
# 2) Greedy policy, epsilon = 0, no exploration. The game is pretty stochastic as it is, no need.
# 3) The valuation function is basically just a linear operator. It takes a vector of the values of
#    1114112 (=65536 * 17) features and dot-product it by the vector of 1114122 weights.
#    Sounds like a lot of computation but! and this is the beauty - all except 17 of the features
#    are exactly zero, and those 17 are exactly 1. So the whole dot product is just a sum of 17 weights,
#    corresponding to the 1-features.
# 4) The same goes for back-propagation. We only need to update 17 numbers of 1m+ on every step.
# 5) But in fact we update 17 * 8 weights using an obvious D4 symmetry group acting on the board


class QAgent:

    max_train_history = 200

    def __init__(self, name: str, user: str, debug=False, no_logs=False):

        # basic params
        self.name = name
        self.user = user
        self.best_game_name = f'best_of_{name}'
        self.top_game = Game(params=BACK.games.find_one({'name': self.best_game_name}))
        self.debug = debug
        if no_logs:
            self.print = no_log_function
        elif debug:
            self.print = print
        else:
            self.print = self.silent_log

        match self.name:
            case 'Random Moves':
                self.evaluate = random_eval
            case 'Best Score':
                self.evaluate = score_eval
            case _:
                self.evaluate = self._evaluate
                agent = BACK.get_agent(self.name)
                self.n = agent['N']
                self.weightSignature = agent['weightSignature']
                self.alpha = agent['alpha']
                self.decay = agent['decay']
                self.step = agent['step']
                self.minAlpha = agent['minAlpha']
                self.maxTile = agent['maxTile']
                self.bestScore = agent['bestScore']
                self.lastTrainingEpisode = agent['lastTrainingEpisode']
                self.history = agent['history']
                self.collectStep = agent['collectStep']
                self.save_agent_keys = ('weightSignature', 'alpha', 'bestScore', 'maxTile',
                                        'lastTrainingEpisode', 'history', 'nextDecay', 'collectStep')
                self.num_feat, self.size_feat = PAR_SHAPE[self.n]
                self.features = FEATURE_FUNCTIONS[self.n]
                self.nextDecay = agent.get('nextDecay', self.lastTrainingEpisode + self.step)

                self.weights = None
                if not self.weightSignature:
                    self.init_weights()
                else:
                    self.load_weights()

    def __str__(self) -> str:
        return f'Agent {self.name}, n={self.n}\ntrained for {self.lastTrainingEpisode} episodes, ' \
               f'top score = {self.bestScore}'

    def silent_log(self, log: str):
        BACK.add_log(self.user, log, add_time=False)

    def init_weights(self):
        if self.n == 6:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist() + \
                           (np.random.random((12, CUTOFF_FOR_6_F ** 6)) / 100).tolist()
            self.weightSignature = [17, 4, 12]
        elif self.n == 5:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist()
            self.weightSignature = [17, 4]
        else:
            self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()
            self.weightSignature = [self.num_feat]

    def load_weights(self):
        self.print('loading weights ...')
        w = BACK.s3_load(self.name)
        if w is None:
            self.init_weights()
        else:
            self.weights = []
            for weight_component in w:
                self.weights += weight_component.tolist()
            del w

    def save_agent(self, with_weights=True):
        if self.name in EXTRA_AGENTS:
            return
        if with_weights:
            start = 0
            nps = []
            for d in self.weightSignature:
                piece = self.weights[start: start + d]
                nps.append(np.array(piece, dtype=np.float32))
                start += d
        else:
            nps = None
        self.maxTile = int(self.maxTile)
        agent_params = {key: getattr(self, key) for key in self.save_agent_keys}
        BACK.save_agent(self.name, agent_params, nps)
        del nps

    def save_game(self, game: Game, game_name: str):
        game.name = game_name
        game.user = game.user or self.user
        BACK.games.replace_one({'name': game_name}, game.to_mongo(), upsert=True)

    def _evaluate(self, row: np.ndarray, score=0) -> float:
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])

    # The numpy library has very nice functions of transpose, rot90, ravel etc.
    # No actual data relocation happens, just the "view" is changed. So it's very fast.
    def update(self, row: np.ndarray, dw: float):
        for _ in range(4):
            for i, f in enumerate(self.features(row)):
                self.weights[i][f] += dw
            row = np.transpose(row)
            for i, f in enumerate(self.features(row)):
                self.weights[i][f] += dw
            row = np.rot90(np.transpose(row))

    # The game 2048 has two kinds of states. After we make a move - this is the one we try to evaluate,
    # and after the random 2-4 tile is placed afterwards.
    # On each step we check which of the available moves leads to a state, which has the highest value
    # according to the current weights of our evaluator. Now we use that best value, our learning rate
    # and the usual Bellman Equation to make a back-propagation update for the previous such state.
    # In this case - we adjust several weights by the same small delta.
    # A very fast and efficient procedure.
    # Then we move in that best direction, add random tile and proceed to the next cycle.
    def episode(self) -> Game:
        game = Game()
        state, old_label = None, 0

        while not game.game_over(game.row):
            action, best_value = 0, -np.inf
            best_row, best_score = None, None
            for direction in range(4):
                new_row, new_score, change = game.pre_move(game.row, game.score, direction)
                if change:
                    value = self.evaluate(new_row)
                    if value > best_value:
                        action, best_value = direction, value
                        best_row, best_score = new_row, new_score
            if state is not None:
                dw = (best_score - game.score + best_value - old_label) * self.alpha / self.num_feat
                self.update(state, dw)
            game.row, game.score = best_row, best_score
            game.numMoves += 1
            game.moves.append(action)
            state, old_label = game.row.copy(), best_value
            game.new_tile()
        game.moves.append(-1)
        dw = - old_label * self.alpha / self.num_feat
        self.update(state, dw)

        return game

    def decay_alpha(self, job_name=''):
        self.alpha = round(max(self.alpha * self.decay, self.minAlpha), 6)
        self.nextDecay = self.lastTrainingEpisode + self.step
        self.print(f'At episode {self.lastTrainingEpisode + 1}, LR decayed to = {round(self.alpha, 6)}')
        BACK.save_new_alpha(job_name, self.alpha)

    def train_run(self, job: dict) -> str:
        user = job['user']
        job_name = job['description']
        global_start = start_1000 = job['start']
        eps = job['episodes']
        first_episode = self.lastTrainingEpisode
        last_episode = self.lastTrainingEpisode + eps
        av1000 = []
        ma100 = []
        ma_collect = []
        reached = [0] * 7
        best_of_1000 = Game()
        self.print(f'{self.name} training started, {eps} episodes\n'
                   f'Agent is saved every 1000 episodes, or on STOP command')

        while self.lastTrainingEpisode < last_episode:
            status = BACK.check_job_status(job_name)
            if status == JobStatus.KILL:
                return f'{job_name} killed by {user}'
            if status == JobStatus.STOP:
                self.save_agent()
                break

            game = self.episode()
            self.lastTrainingEpisode += 1
            ma100.append(game.score)
            av1000.append(game.score)
            max_tile = np.max(game.row)

            if game.score > best_of_1000.score:
                best_of_1000 = game
                if game.score > self.bestScore:
                    self.bestScore = game.score
                    self.top_game = game
                    self.print(f'\nNew best game at episode {self.lastTrainingEpisode}!\n{game.__str__()}')
                    self.save_game(game, self.best_game_name)

            if max_tile >= 10:
                reached[max_tile - 10] += 1
            self.maxTile = max(self.maxTile, max_tile)

            if self.lastTrainingEpisode % 10 == 0 and self.lastTrainingEpisode > 100:
                elapsed_time = time_now() - global_start
                remaining_time = int(elapsed_time / (self.lastTrainingEpisode - first_episode)
                                     * (last_episode - self.lastTrainingEpisode))
                BACK.update_timing(job_name, elapsed_time, remaining_time)

            if self.lastTrainingEpisode % 100 == 0:
                average = np.mean(ma100)
                ma_collect.append(average)
                self.print(f'episode {self.lastTrainingEpisode}, last 100 average = {int(average)}')
                ma100 = []
                if self.lastTrainingEpisode % self.collectStep == 0:
                    self.history.append(int(np.mean(ma_collect)))
                    ma_collect = []
                    if len(self.history) == self.max_train_history:
                        self.history = self.history[1::2]
                        self.collectStep *= 2
                    self.save_agent(with_weights=(self.lastTrainingEpisode % 1000 == 0))

            if self.lastTrainingEpisode % 1000 == 0:
                average = int(np.mean(av1000))
                len_1000 = len(av1000)
                log = f'\n{string_time_now()}: episode = {self.lastTrainingEpisode}\n{time_since(start_1000)} ' \
                      f'for last {len_1000} episodes\naverage score = {average}\n'
                for j in range(7):
                    r = int(sum(reached[j:]) / len_1000 * 10000) / 100
                    if r:
                        log += f'{1 << (j + 10)} reached in {r} %\n'
                log += f'best game of last 1000:\n{best_of_1000.__str__()}\n' \
                       f'best game of Agent:\n{self.top_game.__str__()}\n' \
                       f'{string_time_now()}: {self.name} weights saved'
                av1000 = []
                reached = [0] * 7
                best_of_1000 = Game()
                start_1000 = time_now()
                self.print(log)

            if self.lastTrainingEpisode == self.nextDecay and self.alpha > self.minAlpha:
                self.decay_alpha(job_name)

        self.print('saving weights ...')
        self.save_agent()
        self.print(f'Total time = {time_since(global_start)}')
        return f'{string_time_now()}: {self.name} saved, {self.lastTrainingEpisode} training episodes'

    def test_run(self, job: dict) -> str:
        job_name = job['description']
        user = job['user']
        global_start = start = job['start']
        eps = job['episodes']
        depth = job['depth']
        width = job['width']
        trigger = job['trigger']
        best_trial_name = f'Last_trial_{self.user}'
        self.print(f'{self.name} test session started, {eps} test episodes\n'
                   f'Looking forward: depth = {depth}, width = {width}, trigger = {trigger} empty cells\n')

        results = []
        top_three = [(2, ''), (1, ''), (0, '')]
        for i in range(1, eps + 1):
            status = BACK.check_job_status(job_name)
            if status == JobStatus.KILL:
                return f'{job_name} killed by {user}'
            if status == JobStatus.STOP:
                break

            now = time.time()
            game = Game()
            game.trial_run(estimator=self.evaluate, depth=depth, width=width, trigger=trigger)
            self.print(f'game {i}, score {game.score}, moves {game.numMoves}, reached {get_max_tile(game.row)}, '
                       f'time = {(time.time() - now):.2f} sec')
            score = game.score
            results.append((score, get_max_tile(game.row), game.numMoves))
            if score > top_three[2][0]:
                top_three[2] = (score, game.__str__())
                if score > top_three[1][0]:
                    top_three[2], top_three[1] = top_three[1], top_three[2]
                    if score > top_three[0][0]:
                        top_three[1], top_three[0] = top_three[0], top_three[1]
                        self.save_game(game, best_trial_name)

            elapsed_time = time_now() - start
            if elapsed_time > 5:
                total_elapsed = time_now() - global_start
                remaining_time = int(total_elapsed / i * (eps - i))
                BACK.update_timing(job_name, total_elapsed, remaining_time)
                start = time_now()

        if not results:
            return f'No results collected'
        average = np.average([v[0] for v in results])
        figures = [v[1] for v in results]
        total_odo = sum([v[2] for v in results])

        def share(limit):
            return int(len([0 for v in figures if v >= limit]) / len(figures) * 10000) / 100

        log = '\nBest games:\n'
        for v in top_three:
            if v[1]:
                log += v[1]

        elapsed = time.time() - global_start
        log = f'\naverage score of {len(results)} runs = {average}\n16384 reached in {share(16384)}%\n' \
              f'8192 reached in {share(8192)}%\n4096 reached in {share(4096)}%\n2048 reached in {share(2048)}%\n' \
              f'1024 reached in {share(1024)}%\ntotal time = {time_since(global_start)}\n' \
              f'average time per move = {round(elapsed / total_odo * 1000, 2)} ms\n'
        self.print(log)
        return f'Best game was saved as {best_trial_name}\n--------------'
