from .game_logic import *
from collections import deque


# Argument x for feature functions is a (4, 4) numpy array of np.int32

# features = all adjacent pairs
def f_2(x):
    x_vert = ((x[:3, :] << 4) + x[1:, :]).ravel()
    x_hor = ((x[:, :3] << 4) + x[:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor])


# features = all adjacent triples, i.e. 3 in a row + 3 in a any square missing one corner
def f_3(x):
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
def f_4(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    return np.concatenate([x_vert, x_hor, x_sq])


# Finally, we try adding 4 "cross" 5-features for middle cells
def f_5(x):
    x_vert = ((x[0, :] << 12) + (x[1, :] << 8) + (x[2, :] << 4) + x[3, :]).ravel()
    x_hor = ((x[:, 0] << 12) + (x[:, 1] << 8) + (x[:, 2] << 4) + x[:, 3]).ravel()
    x_sq = ((x[:3, :3] << 12) + (x[1:, :3] << 8) + (x[:3, 1:] << 4) + x[1:, 1:]).ravel()
    x_middle = ((x[1: 3, 1: 3] << 16) + (x[:2, 1: 3] << 12) + (x[1: 3, :2] << 8) + (x[2:, 1: 3] << 4) + x[1: 3, 2:]
                ).ravel()
    return np.concatenate([x_vert, x_hor, x_sq, x_middle])


# Adding some limited 6-features, up to < 2 ** (cutoff - 1) > tile
def f_6(x):
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

    def __init__(self, name: str, job_idx: str, idx: str, debug=False, no_logs=False):

        # basic params
        self.debug = debug
        self.name = name
        self.job_idx = job_idx
        self.idx = idx
        self.best_game_idx = f'best_of_{idx}'
        self.best_trial_idx = f'last_trial_{idx}'

        if no_logs:
            self.print = no_log_function
        elif debug:
            self.print = print
        else:
            self.print = self.silent_log

        self.save_agent_keys = ('weight_signature', 'alpha', 'best_score', 'max_tile', 'train_eps',
                                'train_history', 'collect_step')
        self.top_game = Game(params=BACK.get_game(self.best_game_idx))
        self.best_score = self.top_game.score

        match self.idx:
            case 'Random Moves':
                self.evaluate = random_eval
            case 'Best Score':
                self.evaluate = score_eval
            case _:
                # agent params from Database
                agent = BACK.get_agent(self.idx)
                self.n = agent['n']
                self.weight_signature = agent['weight_signature']
                self.alpha = agent['alpha']
                self.decay = agent['decay']
                self.step = agent['step']
                self.min_alpha = agent['min_alpha']
                self.max_tile = agent['max_tile']
                self.train_eps = agent['train_eps']
                self.train_history = agent['train_history']
                self.collect_step = agent['collect_step']

                # derived params
                self.num_feat, self.size_feat = PAR_SHAPE[self.n]
                self.features = FEATURE_FUNCTIONS[self.n]

                # operational params
                self.best_game = None
                self.next_decay = self.train_eps + self.step
                self.trigger_tile = 10

                self.weights = None
                if self.weight_signature is None:
                    self.init_weights()
                else:
                    self.load_weights()

                self.evaluate = self._evaluate

    def __str__(self):
        return f'Agent {self.idx}, n={self.n}\ntrained for {self.train_eps} episodes, top score = {self.best_score}'

    def init_weights(self):
        if self.n == 6:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist() + \
                           (np.random.random((12, CUTOFF_FOR_6_F ** 6)) / 100).tolist()
            self.weight_signature = [17, 4, 12]
        elif self.n == 5:
            self.weights = (np.random.random((17, 16 ** 4)) / 100).tolist() + \
                           (np.random.random((4, 16 ** 5)) / 100).tolist()
            self.weight_signature = [17, 4]
        else:
            self.weights = (np.random.random((self.num_feat, self.size_feat)) / 100).tolist()
            self.weight_signature = [self.num_feat]

    def load_weights(self):
        self.print('loading weights ...')
        w = BACK.s3_load(self.idx)
        self.weights = []
        for weight_component in w:
            self.weights += weight_component.tolist()
        del w

    def silent_log(self, log):
        BACK.add_log(self.name, log)

    def save_agent(self, with_weights=True):
        if self.idx in EXTRA_AGENTS:
            return
        if with_weights:
            start = 0
            nps = []
            for d in self.weight_signature:
                piece = self.weights[start: start + d]
                nps.append(np.array(piece, dtype=np.float32))
                start += d
        else:
            nps = None
        self.max_tile = int(self.max_tile)
        agent_params = {key: getattr(self, key) for key in self.save_agent_keys}
        BACK.save_agent(self.idx, agent_params, nps)
        del nps

    def save_game(self, game: Game, idx: str):
        if self.idx in EXTRA_AGENTS:
            return
        game.idx = idx
        game.player = game.player or f'Agent {self.idx}'
        BACK.save_game(self.name, game.to_dict())

    def _evaluate(self, row, score=None):
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])

    # The numpy library has very nice functions of transpose, rot90, ravel etc.
    # No actual data relocation happens, just the "view" is changed. So it's very fast.
    def update(self, row, dw):
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
    def episode(self):
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
            game.n_moves += 1
            game.moves.append(action)
            state, old_label = game.row.copy(), best_value
            game.new_tile()
        game.moves.append(-1)
        dw = - old_label * self.alpha / self.num_feat
        self.update(state, dw)

        return game

    def decay_alpha(self):
        self.alpha = round(max(self.alpha * self.decay, self.min_alpha), 4)
        self.next_decay = self.train_eps + self.step
        self.print(f'At episode = {self.train_eps + 1} current learning rate = {round(self.alpha, 4)}')

    def train_run(self, params: dict):
        eps = params['episodes']
        last_episode = self.train_eps + eps
        av1000 = []
        ma_collect = deque(maxlen=self.collect_step)
        reached = [0] * 7
        best_of_1000 = Game()
        global_start = start_1000 = time.time()
        self.print(f'Agent {self.idx} train session started, training episodes = {eps}')
        self.print('Agent will be saved every 1000 episodes and on STOP JOB command')

        while self.train_eps < last_episode:
            # check job status
            status = BACK.check_job_status(self.job_idx)
            if status == -1:
                return f'{time_now()}: Job killed by {self.name}\n--------------'
            if status == 0:
                self.save_agent()
                self.print(f'Job stopped by {self.name}')
                break

            # check if it's time to decay learning rate
            if self.train_eps > self.next_decay and self.alpha > self.min_alpha:
                self.decay_alpha()

            game = self.episode()
            self.train_eps += 1

            ma_collect.append(game.score)
            av1000.append(game.score)
            max_tile = np.max(game.row)

            if game.score > best_of_1000.score:
                best_of_1000 = game
                if game.score > self.best_score:
                    self.best_score = game.score
                    self.top_game = game
                    self.print(f'\nNew best game at episode {self.train_eps}!\n{game.__str__()}\n')
                    self.save_game(game, self.best_game_idx)

            if max_tile >= 10:
                reached[max_tile - 10] += 1

            if max_tile > self.max_tile:
                self.max_tile = max_tile
                if self.max_tile > self.trigger_tile:
                    self.decay_alpha()

            if self.train_eps % 100 == 0:
                ma = int(np.mean(ma_collect))
                self.print(f'episode {self.train_eps}: score {game.score}, reached {1 << max_tile},'
                           f' ma_{self.collect_step} = {ma}')
                if self.train_eps % self.collect_step == 0:
                    self.train_history.append(ma)
                    if len(self.train_history) == 200:
                        self.train_history = self.train_history[1::2]
                        self.collect_step *= 2
                    self.save_agent(with_weights=False)

            if self.train_eps % 1000 == 0:
                average = int(np.mean(av1000))
                len_1000 = len(av1000)
                self.print(f'\n{time_now()}: episode = {self.train_eps}')
                self.print(f'{lapse_format(start_1000)} for last {len_1000} episodes')
                self.print(f'average score = {average}')
                for j in range(7):
                    r = int(sum(reached[j:]) / len_1000 * 10000) / 100
                    if r:
                        self.print(f'{1 << (j + 10)} reached in {r} %')
                self.print(f'best game of last 1000:')
                self.print(best_of_1000.__str__())
                self.print(f'best game of Agent:')
                self.print(self.top_game.__str__())
                av1000 = []
                reached = [0] * 7
                best_of_1000 = Game()
                self.save_agent()
                self.print(f'{time_now()}: Agent {self.idx} weights saved\n')
                start_1000 = time.time()

        self.print(f'\nTotal time = {lapse_format(global_start)}')
        self.save_agent()
        return f'{time_now()}: Agent {self.idx} saved, {self.train_eps} training episodes\n------------------------'

    def test_run(self, params):
        eps = params['episodes']
        depth = params['depth']
        width = params['width']
        trigger = params['trigger']
        start = time.time()
        self.print(f'Test run of {eps} episodes for Agent {self.idx}\n'
                   f'Looking forward with: depth = {depth}, width = {width}, trigger = {trigger} empty cells')

        results = []
        for i in range(eps):
            # check job status
            if not self.debug:
                status = BACK.check_job_status(self.job_idx)
                if status == -1:
                    return f'{time_now()}: Job killed by {self.name}'
                if status == 0:
                    self.save_agent()
                    self.print(f'Job stopped by {self.name}')
                    break

            now = time.time()
            game = Game()
            game.trial_run(estimator=self.evaluate, depth=depth, width=width, trigger=trigger)
            self.print(f'game {i}, result {game.score}, moves {game.n_moves}, achieved {get_max_tile(game.row)}, '
                       f'time = {(time.time() - now):.2f}')
            results.append(game)

        if not results:
            return f'No results collected\n--------------'
        average = np.average([v.score for v in results])
        figures = [get_max_tile(v.row) for v in results]
        total_odo = sum([v.n_moves for v in results])
        results.sort(key=lambda v: v.score, reverse=True)

        def share(limit):
            return int(len([0 for v in figures if v >= limit]) / len(figures) * 10000) / 100

        self.print('\nBest games:\n')
        for v in results[:3]:
            self.print(f'{v.__str__()}\n')
        elapsed = time.time() - start
        self.print(f'average score of {len(results)} runs = {average}\n'
                   f'16384 reached in {share(16384)}%\n' + f'8192 reached in {share(8192)}%\n'
                   f'4096 reached in {share(4096)}%\n' + f'2048 reached in {share(2048)}%\n'
                   f'1024 reached in {share(1024)}%\n' + f'total time = {round(elapsed, 2)}\n'
                   f'average time per move = {round(elapsed / total_odo * 1000, 2)} ms\n')
        self.save_game(results[0], self.best_trial_idx)
        return f'Best game was saved as {self.best_trial_idx}\n--------------'
