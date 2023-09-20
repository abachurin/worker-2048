from base.watch_agent import *

JOBS = {}


def worker(job: dict):
    user = job['user']

    STOPPER['check_memory'] = False
    Thread(target=get_top_memory, args=(job['description'], )).start()

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

        STOPPER['check_memory'] = True

        if status:
            BACK.add_log(user, status + '\n')
    except Exception as ex:
        BACK.add_log('admin', f'{job["description"]} failed: {str(ex)}')
        BACK.add_log(user, f'{job["description"]} failed: {str(ex)}\n')
    finally:
        BACK.delete_job(job['description'])


def main():
    clean_temp_dir()
    BACK.clean_watch_jobs()
    BACK.admin_full_update(len(JOBS))
    current_jobs = 0
    error = False
    while True:
        try:
            active, pending = BACK.active_jobs()
            close_workers = [v for v in JOBS if v not in active]
            open_workers = [v for v in pending if v not in JOBS]
            for job_name in close_workers:
                psutil.Process(JOBS[job_name]).terminate()
                user = job_name.split()[-1]
                BACK.add_log(user, f'{job_name} is over\n*****')
                del JOBS[job_name]
                print(f'{job_name}: over')
            for job_name in open_workers:
                job = BACK.launch_job(job_name)
                if job is None:
                    print(f"{job_name} doesn't exist")
                    continue
                p = Process(target=worker, args=(job, ), daemon=True)
                p.start()
                JOBS[job_name] = p.pid
                BACK.add_log('admin', f'{job_name}')
            if not active:
                clean_temp_dir()
            if error:
                BACK.add_log('admin', 'Worker back online')
                error = False
        except Exception as ex:
            print(ex)
            BACK.add_log('admin', str(ex))
            error = True
        finally:
            time.sleep(3)
            if len(JOBS) != current_jobs:
                BACK.clean_watch_games()
                BACK.admin_full_update(len(JOBS))
                current_jobs = len(JOBS)


if __name__ == '__main__':

    main()
