from datetime import datetime
import os
import json
import time
from pprint import pprint
import boto3
from pymongo import MongoClient
import pickle
import random
import numpy as np
from enum import Enum
import shutil
from pathlib import Path

TMP_DIR = os.path.join(os.getcwd(), 'tmp', '')


def clean_temp_dir():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)


clean_temp_dir()


def full_key(name):
    return f'{name}.pkl'


def time_suffix():
    return str(datetime.now())[-6:]


def temp_local():
    return os.path.join(TMP_DIR, f'tmp{time_suffix()}.pkl')


def time_now():
    return str(datetime.now())[:19]


def lapse_format(t):
    diff = int(time.time() - t)
    return f'{diff // 60} min {diff % 60} sec'


def no_log_function(text):
    return


class Backend:

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
        cluster = f'mongodb+srv://{mongo_creds["user"]}:{mongo_creds["pwd"]}@instance-0' \
                  f'.55byx.mongodb.net/?retryWrites=true&w=majority'
        client = MongoClient(cluster)
        db = client[mongo_creds['db']]
        self.users = db['users']

    def s3_files(self):
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

    def s3_load(self, name: str):
        key = full_key(name)
        if key not in self.s3_files():
            return
        temp = temp_local()
        self.space.download_file(key, temp)
        with open(temp, 'rb') as f:
            result = pickle.load(f)
        os.remove(temp)
        return result

    def active_users(self):
        return [v['name'] for v in self.users.find({'Jobs': {'$ne': []}}, {'name': 1, '_id': 0})]

    def get_item(self, idx: str, kind: str):
        try:
            return self.users.find_one({f'{kind}.idx': idx}, {f'{kind}.$': 1})[kind][0]
        except TypeError:
            return None

    def replace_item(self, name: str, item: dict, kind: str):
        self.users.update_one({'name': name}, {'$pull': {kind: {'idx': item['idx']}}})
        self.users.update_one({'name': name}, {'$push': {kind: item}})

    def get_game(self, idx: str):
        return self.get_item(idx, 'Games')

    def get_agent(self, idx: str):
        return self.get_item(idx, 'Agents')

    def save_game(self, name: str, game: dict):
        self.replace_item(name, game, 'Games')

    def save_agent(self, idx: str, update: dict, weights: object):
        for key in update:
            self.users.update_one({'Agents.idx': idx}, {'$set': {f'Agents.$.{key}': update[key]}})
        self.s3_save(weights, idx)

    def add_log(self, name: str, log: str):
        self.users.update_one({'name': name}, {'$push': {'logs': log}})

    def get_first_job(self, name: str):
        try:
            return self.users.find_one({'name': name}, {'Jobs': 1})['Jobs'][0]
        except Exception:
            return None

    def check_job_status(self, idx: str):
        try:
            return self.get_item(idx, 'Jobs')['status']
        except TypeError:
            return -1

    def set_job_status(self, idx: str, new_status: int):
        self.users.update_one({'Jobs.idx': idx}, {'$set': {'Jobs.$.status': new_status}})

    def launch_job(self, name: str, idx: str, t: str):
        self.users.update_one({'Jobs.idx': idx}, {'$set': {'Jobs.$.launch_time': t, 'Jobs.$.status': 2}})
        self.add_log(name, f'{t}: {idx} launched')

    def delete_job(self, idx: str):
        self.users.update_one({'Jobs.idx': idx}, {'$pull': {'Jobs': {'idx': idx}}})

    def delete_watch_user(self, idx: str):
        self.users.delete_one({'Jobs.idx': idx})

    # related to Watch Agent operations
    def start_watch_job(self, idx):
        self.users.update_one({'Jobs.idx': idx}, {'$set': {'Jobs.$.new_game': 0}})

    def new_watch_job(self, idx):
        try:
            return self.users.find_one({'Jobs.idx': idx}, {'Jobs.$': 1})['Jobs'][0]['new_game']
        except TypeError:
            return -1

    def update_watch_job(self, idx: str, moves: list, tiles: list):
        self.users.update_one({'Jobs.idx': idx}, {
            '$push': {
                'Jobs.$.moves': {'$each': moves},
                'Jobs.$.tiles': {'$each': tiles}},
            '$set': {
                'Jobs.$.new_game': 0}})

    def clean_watch_jobs(self):
        self.users.delete_many({'status': 'tmp'})


with open('base/config.json', 'r') as f:
    CONF = json.load(f)
LOCAL = os.environ.get('S3_URL', 'local') == 'local'

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
        'db': os.getenv('MG_DB', 'robot-2048'),
    }


BACK = Backend(s3_credentials, mongo_credentials)
