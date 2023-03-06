from base.watch_agent import *
import psutil
from multiprocessing import Process

WORKERS = {
}


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
                match job['mode']:
                    case 'train':
                        status = QAgent(name=job['name'], job_idx=job['idx'], idx=job['agent']).train_run(job)
                    case 'test':
                        status = QAgent(name=job['name'], job_idx=job['idx'], idx=job['agent']).test_run(job)
                    case 'watch':
                        status = watch_run(job)
                        BACK.delete_watch_user(idx)
                    case _:
                        status = None
                if status:
                    BACK.add_log(name, status + '\n')
            except Exception as ex:
                BACK.add_log(name, f'{time_now()}: Job {idx} failed: {str(ex)}\n')
            BACK.delete_job(idx)
        else:
            time.sleep(1)


def main():
    BACK.clean_watch_jobs()
    while True:
        active = BACK.active_users()
        close_workers = [v for v in WORKERS if v not in active]
        open_workers = [v for v in active if v not in WORKERS]
        for name in close_workers:
            psutil.Process(WORKERS[name]).terminate()
            BACK.add_log(name, f'{time_now()}: No working jobs for {name}\n*****\n')
            del WORKERS[name]
            print(f'kill {name}')
        for name in open_workers:
            p = Process(target=worker, args=(name, ), daemon=True)
            p.start()
            WORKERS[name] = p.pid
            print(f'start {name}')
        if not active:
            clean_temp_dir()
        time.sleep(3)


if __name__ == '__main__':

    main()
