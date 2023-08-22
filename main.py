from base.watch_agent import *
import psutil
from multiprocessing import Process

WORKERS = {}


def log_memory_usage():
    memo = psutil.virtual_memory()
    mb = 1 << 20
    BACK.add_log('Loki', f'{str(datetime.now())[:19]} | total: {int(memo.total / mb)} '
                         f'| used: {int(memo.used / mb)} | available: {int(memo.available / mb)}')


def worker(job: dict):
    user = job['user']
    try:
        match job['type']:
            case JobType.TRAIN.value:
                status = QAgent(name=job['name'], user=user).train_run(job)
            case JobType.TEST.value:
                status = QAgent(name=job['name'], user=user).test_run(job)
            case JobType.WATCH.value:
                status = watch_run(job)
            case _:
                status = None
        if status:
            BACK.add_log(user, status + '\n')
    except Exception as ex:
        BACK.add_log(user, f'{string_time_now()}: {job["description"]} failed: {str(ex)}\n')
    finally:
        BACK.delete_job(job["description"])


def main():
    clean_temp_dir()
    BACK.clean_watch_jobs()
    save_memory_counter = 0
    while True:
        try:
            active, pending = BACK.active_jobs()
            close_workers = [v for v in WORKERS if v not in active]
            open_workers = [v for v in pending if v not in WORKERS]
            for job_name in close_workers:
                psutil.Process(WORKERS[job_name]).terminate()
                user = job_name.split()[-1]
                BACK.add_log(user, f'{string_time_now()}: {job_name} is over\n*****\n')
                del WORKERS[job_name]
                print(f'{job_name}: over')
            for job_name in open_workers:
                job = BACK.launch_job(job_name)
                if job is None:
                    print(f"strange ... the {job_name} doesn't exist")
                    continue
                p = Process(target=worker, args=(job, ), daemon=True)
                p.start()
                WORKERS[job_name] = p.pid
                print(f'start: {job_name}')
            if not active:
                clean_temp_dir()
            if save_memory_counter % 3600 == 0:
                log_memory_usage()
        except Exception as ex:
            print(f'{time_now()}: {str(ex)}')
        finally:
            time.sleep(3)
            save_memory_counter += 1


if __name__ == '__main__':

    main()
