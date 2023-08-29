from datetime import datetime, timedelta
import os
import json
import time
from collections import deque
import re
from pprint import pprint
import boto3
from pymongo import MongoClient
import pickle
import random
import numpy as np
from enum import Enum
import shutil
from pathlib import Path
from typing import List, Union, Mapping, Any, Tuple, Callable, Dict
from threading import Thread
from multiprocessing import Process
import psutil


class JobType(Enum):
    TRAIN = 0
    TEST = 1
    WATCH = 2


class JobStatus(Enum):
    PENDING = 0
    RUN = 1
    STOP = 2
    KILL = 3


class NewGameJob(Enum):
    KILL = 0
    RESTART = 1
    KEEP = 2


TMP_DIR = os.path.join(os.getcwd(), 'tmp', '')
WAIT_TO_CONTINUE_WATCH = 300
SAVE_NEW_MOVES = 2
STOPPER: Dict[str, bool] = {}


def clean_temp_dir():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)


def full_key(name: str) -> str:
    return f'{name}.pkl'


def time_suffix() -> str:
    return str(datetime.now())[-6:]


def temp_local() -> str:
    return os.path.join(TMP_DIR, f'tmp{time_suffix()}.pkl')


def time_now() -> int:
    return int(time.time())


def time_since(t: float) -> str:
    return str(timedelta(seconds=time_now() - int(t)))


def string_from_ts(ts: int):
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def string_time_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def no_log_function(_):
    return


class Backend:

    watch_game_pattern = {"$regex": r'^\*'}
    admin_logs_limit = 200

    def __init__(self, s3_creds: dict, mongo_creds: dict):
        # S3 Storage
        qwargs = {
            'service_name': 's3',
            'endpoint_url': f'https://{s3_creds["region"]}.digitaloceanspaces.com',
            'region_name': s3_creds['region'],
            'aws_access_key_id': s3_creds['access_key'],
            'aws_secret_access_key': s3_creds['secret_key']
        }
        self.engine = boto3.resource(**qwargs)
        self.client = boto3.client(**qwargs)
        self.space_name = s3_creds['space']
        self.space = self.engine.Bucket(self.space_name)
        # MongoDB
        cluster = f'mongodb+srv://{mongo_creds["user"]}:{mongo_creds["pwd"]}@{mongo_creds["location"]}'
        client = MongoClient(cluster)
        self.db = client[mongo_creds['db']]
        self.users = self.db['users']
        self.agents = self.db['agents']
        self.games = self.db['games']
        self.jobs = self.db['jobs']
        self.workers = {}

    # Memory management
    @staticmethod
    def memory_used() -> int:
        return psutil.virtual_memory().used >> 20

    @staticmethod
    def memory_free() -> int:
        return psutil.virtual_memory().available >> 20

    def s3_used_space(self) -> int:
        return sum([o.size for o in self.space.objects.all()]) >> 20

    def mongo_used_space(self) -> int:
        return int(self.db.command('dbstats')['totalSize']) >> 20

    def update_admin(self, fields: dict):
        self.users.update_one({'name': 'admin'}, {'$set': fields})

    def admin_logs(self, log:  str):
        admin = self.users.find_one({'name': 'admin'}, {'logs': 1})
        if admin is not None:
            logs = admin['logs'] + log.split('\n')
            if len(logs) > self.admin_logs_limit + 50:
                logs = logs[-self.admin_logs_limit:]
            self.update_admin({'logs': logs})

    def admin_update(self):
        memo_used = self.memory_used()
        memo_free = self.memory_free()
        s3_used = self.s3_used_space()
        mongo_used = self.mongo_used_space()
        self.update_admin({'memoUsed': memo_used, 'memoFree': memo_free, 's3Used': s3_used, 'mongoUsed': mongo_used})

    def admin_adjust_memo(self, job_name: str):
        admin = self.users.find_one({'name': 'admin'}, {'memoProjected': 1})
        job = self.jobs.find_one({'description': job_name}, {'memoProjected': 1})
        if admin and job:
            self.jobs.update_one({'description': job_name}, {'$set': {'memoProjected': 0}})
            self.users.update_one({'name': 'admin'},
                                  {'$set': {'memoProjected': admin['memoProjected'] - job['memoProjected']}})
            self.admin_update()

    # S3 Storage
    def s3_files(self) -> List[str]:
        return [o.key for o in self.space.objects.all()]

    def s3_save(self, data: object, idx: str):
        if data is None:
            return
        temp = temp_local()
        key = full_key(idx)
        with open(temp, 'wb') as f:
            pickle.dump(data, f, -1)
        self.space.upload_file(temp, key)
        os.remove(temp)

    def s3_load(self, name: str) -> object:
        key = full_key(name)
        if key not in self.s3_files():
            return
        temp = temp_local()
        self.space.download_file(key, temp)
        with open(temp, 'rb') as f:
            result = pickle.load(f)
        os.remove(temp)
        return result

    # MongoDB

    # General Job management
    def active_jobs(self) -> Tuple[List[str], List[str]]:
        jobs = [v for v in self.jobs.find({}, {'description': 1, 'status': 1})]
        active = [v['description'] for v in jobs]
        pending = [v['description'] for v in jobs if v['status'] == JobStatus.PENDING.value]
        return active, pending

    def launch_job(self, job_name: str) -> Union[None, Mapping[str, Any]]:
        self.jobs.update_one({'description': job_name},
                             {'$set': {'start': time_now(), 'status': JobStatus.RUN.value}})
        job = self.jobs.find_one({'description': job_name})
        if job is None:
            return None
        if job['type'] != JobType.WATCH.value:
            self.add_log(job['user'], f'\n{string_time_now()}: {job_name} launched')
        return job

    def delete_job(self, job_name: str):
        job = self.jobs.find_one({'description': job_name}, {'memoProjected': 1})
        if job:
            admin = self.users.find_one({'name': 'admin'}, {'memoProjected': 1})
            if admin:
                self.users.update_one({'name': 'admin'},
                                      {'$set': {'memoProjected': admin['memoProjected'] - job['memoProjected']}})
            self.jobs.delete_one({'description': job_name})

    def check_job_status(self, job_name: str) -> JobStatus:
        job = self.jobs.find_one({'description': job_name})
        if job is None:
            return JobStatus.KILL
        return JobStatus(job['status'])

    def add_log(self, name: str, log: str):
        lines = log.split('\n')
        self.users.update_one({'name': name}, {'$push': {'logs': {'$each': lines}}})

    # Train/Test Agent
    def update_timing(self, job_name: str, elapsed_time: int, remaining_time: str):
        self.jobs.update_one({'description': job_name},
                             {'$set': {'timeElapsed': elapsed_time, 'remainingTimeEstimate': remaining_time}})

    def save_new_alpha(self, job_name: str, alpha: float):
        self.jobs.update_one({'description': job_name}, {'$set': {'alpha': alpha}})

    def get_agent(self, agent_name: str) -> Union[None, Mapping[str, Any]]:
        return self.agents.find_one({'name': agent_name})

    def save_agent(self, name: str, update: dict, weights: object):
        self.agents.update_one({'name': name}, {'$set': update})
        self.s3_save(weights, name)

    # Watch Agent
    def clean_watch_jobs(self):
        self.jobs.delete_many({'type': JobType.WATCH.value})
        self.games.delete_many({'name': self.watch_game_pattern})

    def clean_watch_games(self):
        watch_users = [v['user'] for v in self.jobs.find({'type': JobType.WATCH.value})]
        self.games.delete_many({'user': {"$nin": watch_users}})
        all_games = [v['name'] for v in self.games.find()]
        for name in STOPPER:
            if name not in all_games:
                del STOPPER[name]

    def get_game_params(self, game_name: str) -> Union[None, Mapping[str, Any]]:
        return self.games.find_one({'name': game_name})

    def save_game(self, name: str, game: dict):
        self.games.replace_one({'name': name}, game, upsert=True)

    # related to Watch Agent operations
    def launch_watch_job(self, user: str):
        self.jobs.update_one({'description': user}, {'$set': {'status': JobStatus.RUN.value, 'loadingWeights': False}})

    def set_watch_job(self, user: str, status: bool):
        self.jobs.update_one({'description': user}, {'$set': {'loadingWeights': status}})

    def get_watch_game(self, user: str) -> dict:
        job = self.jobs.find_one({'description': user}, {'newGame': 1, 'startGame': 1})
        if job is None:
            self.games.delete_many({'user': user})
            return {'status': NewGameJob.KILL}
        if job['newGame']:
            self.jobs.update_one({'description': user}, {'$set': {'newGame': False}})
            return {**job['startGame'], 'user': user, 'status': NewGameJob.RESTART}
        return {'status': NewGameJob.KEEP}

    def update_watch_game(self, name: str, moves: list, tiles: list):
        self.games.update_one({'name': name}, {
            '$push': {
                'moves': {'$each': moves},
                'tiles': {'$each': tiles}}
            }
        )


with open('base/config.json', 'r') as f:
    CONF = json.load(f)
LOCAL = os.environ.get('AT_HOME', 'local') == 'local'

if LOCAL:
    with open(CONF['s3_credentials'], 'r') as f:
        s3_credentials = json.load(f)
    with open(CONF['mongo_credentials'], 'r') as f:
        mongo_credentials = json.load(f)
else:
    s3_credentials = {
        'region': os.getenv('S3_REGION', None),
        'space': os.getenv('S3_SPACE', 'robot-2048'),
        'access_key': os.getenv('S3_ACCESS_KEY', None),
        'secret_key': os.getenv('S3_SECRET_KEY', None)
    }
    mongo_credentials = {
        'user': os.getenv('MG_USER', None),
        'pwd': os.getenv('MG_PWD', None),
        'location': os.getenv('MG_LOCATION', None),
        'db': os.getenv('MG_DB', 'robot-2048'),
    }


BACK = Backend(s3_credentials, mongo_credentials)
