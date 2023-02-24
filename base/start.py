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


def full_key(name):
    return f'{name}.pkl'


def time_suffix():
    return str(datetime.now())[-6:]


def temp_local():
    return f'tmp{time_suffix()}.pkl'


def time_now():
    return str(datetime.now())[:19]


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

    def s3_save(self, data: object, name: str):
        temp = temp_local()
        key = full_key(name)
        with open(temp, 'w') as f:
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

    def replace_item(self, name: str, item: dict, kind: str):
        self.users.update_one({'name': name}, {'$pull': {kind: {'idx': item['idx']}}})
        self.users.update_one({'name': name}, {'$push': {kind: item}})

    def save_game(self, name: str, game: dict):
        self.replace_item(name, game, 'Games')

    def save_agent(self, name: str, agent: dict, weights: object):
        self.replace_item(name, agent, 'Agents')
        self.s3_save(weights, agent['idx'])

    def add_log(self, name: str, log: str):
        self.users.update_one({'name': name}, {'$push': {'logs': log}})

    def get_first_job(self, name: str):
        cur = self.users.find_one({'name': name}, {'Jobs': 1, '_id': 0})
        if cur and cur['Jobs']:
            return cur['Jobs'][0]
        return None

    def check_job_status(self, idx: str):
        try:
            return self.users.find_one({'Jobs.idx': idx}, {'Jobs.status': 1, '_id': 0})['Jobs'][0]['status']
        except TypeError:
            return 'kill'

    def launch_job(self, name: str, idx: str, now: str):
        self.users.update_one({'Jobs.idx': idx}, {'$set': {'Jobs.$.launch': now}})
        self.add_log(name, f'{now}: {idx} launched')

    def delete_job(self, idx: str):
        self.users.update_one({}, {'$pull': {'Jobs': {'idx': idx}}})


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
