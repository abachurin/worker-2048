from .r_learning import *

WAIT_TO_CONTINUE = 300


class QAgentRun:

    def __init__(self, job_idx: str, agent_idx: str, no_logs=True):
        self.job_idx = job_idx
        self.idx = agent_idx
        self.weights = None
        self.features = None

        self.print = no_log_function if no_logs else print
        match self.idx:
            case 'Random Moves':
                self.evaluate = random_eval
            case 'Best Score':
                self.evaluate = score_eval
            case _:
                n = BACK.get_agent(self.idx)['n']
                self.features = FEATURE_FUNCTIONS[n]
                self.evaluate = self._evaluate
                self.load_weights()

    def load_weights(self):
        self.print('loading weights ...')
        w = BACK.s3_load(self.idx)
        self.weights = []
        for weight_component in w:
            self.weights += weight_component.tolist()
        del w

    def _evaluate(self, row, score=None):
        return sum([self.weights[i][f] for i, f in enumerate(self.features(row))])


def watch_run(job: dict):
    idx = job['idx']
    agent = job['agent']
    estimator = QAgentRun(job_idx=idx, agent_idx=agent).evaluate
    BACK.set_job_status(idx, 3)
    check_sleep = time.time() + WAIT_TO_CONTINUE
    while True:
        job = BACK.get_item(idx, 'Jobs')
        if job is None:
            return
        if job['new_game']:
            params = {
                'row': job['row'],
                'score': job['score'],
                'odo': job['odo']
            }
            game = Game(params=params)
            game.watch_run(idx, estimator, depth=job['depth'], width=job['width'], trigger=job['trigger'])
            check_sleep = time.time() + WAIT_TO_CONTINUE
        else:
            time.sleep(1)
            if time.time() > check_sleep:
                return
