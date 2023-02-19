from .start import *
import boto3
from pymongo import MongoClient
import pickle


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

    def list_files(self):
        return [o.key for o in self.space.objects.all()]

    def save(self, data: object, name: str, kind='Games'):
        temp = temp_local()
        key = full_s3_key(name, kind)
        with open(temp, 'w') as f:
            pickle.dump(data, f, -1)
        self.space.upload_file(temp, key)
        os.remove(temp)

    def load(self, name: str, kind='Agents'):
        key = full_s3_key(name, kind)
        if key not in self.list_files():
            return
        temp = temp_local()
        self.space.download_file(key, temp)
        with open(temp, 'rb') as f:
            result = pickle.load(f)
        os.remove(temp)
        return result

    def find_user(self, name: str):
        user = self.users.find_one({'name': name})
        if user is not None:
            del user['_id']
        return user

    def add_array_item(self, name: str, item: dict, kind='Games'):
        self.users.update_one({'name': name}, {'$addToSet': {kind: item}})

    def add_log(self, name: str, log: str):
        self.users.update_one({'name': name}, {'$push': {'logs': log}})

    def check_job_status(self, idx: str):
        try:
            return self.users.find_one({'Jobs.idx': idx}, {'Jobs.$': 1})['Jobs'][0]['status']
        except TypeError:
            return -1

    def delete_job(self, name: str, idx: str):
        self.users.update_one({'name': name}, {'$pull': {'Jobs': {'idx': idx}}})
