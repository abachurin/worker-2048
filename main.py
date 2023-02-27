from base.r_learning import *
import psutil
from multiprocessing import Process

# {'idx': 'A1', 'n': 4, 'alpha': 0.25, 'decay': 0.75, 'step': 10000, 'min_alpha': 0.01, 'episodes': 100}
#


def worker(name):
    while True:
        job = BACK.get_first_job(name)
        if job is not None:
            idx = job['idx']
            if job['status'] < 1:
                BACK.delete_job(idx)
            now = time_now()
            BACK.launch_job(name, idx, now)
            try:
                agent = QAgent(job, debug=False)
                if job['mode'] == 'train':
                    func = agent.train_run
                else:
                    func = agent.test_run
                status = func(job)
                BACK.add_log(name, status + '\n')
            except Exception as ex:
                BACK.add_log(name, f'{time_now()}: Job {idx} failed: {str(ex)}\n')
            BACK.delete_job(idx)
        else:
            time.sleep(1)


WORKERS = {
}


def main():
    while True:
        active = BACK.active_users()
        close_workers = [v for v in WORKERS if v not in active]
        open_workers = [v for v in active if v not in WORKERS]
        for name in close_workers:
            psutil.Process(WORKERS[name]).terminate()
            del WORKERS[name]
            print(f'kill {name}')
        for name in open_workers:
            p = Process(target=worker, args=(name, ), daemon=True)
            p.start()
            WORKERS[name] = p.pid
            print(f'start {name}')
        time.sleep(2)


if __name__ == '__main__':

    main()

