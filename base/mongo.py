from pymongo import MongoClient


class Mongo:

    def __init__(self, credentials: dict):
        self.cluster = f'mongodb+srv://{credentials["user"]}:{credentials["pwd"]}@instance-0' \
                       f'.55byx.mongodb.net/?retryWrites=true&w=majority'
        self.client = MongoClient(self.cluster)
        self.db = self.client['robot-2048']
        self.users = self.db['users']


    def find_user(self, name: str):
        user = self.users.find_one({'name': name})
        if user is not None:
            del user['_id']
        return user

    def update_user(self, name: str, fields: dict):
        self.users.update_one({'name': name}, {'$set': fields})

    def delete_array_item(self, idx: str, kind: str):
        return self.users.update_one({}, {'$pull': {kind: {'idx': idx}}}).modified_count

    def add_array_item(self, name: str, item: dict, kind: str):
        if item['idx'] in self.all_items(kind):
            return False
        self.users.update_one({'name': name}, {'$push': {kind: item}})
        return True

    def add_log(self, name: str, log: str):
        self.users.update_one({'name': name}, {'$push': {'logs': log}})

    def clear_logs(self, name: str):
        self.users.update_one({'name': name}, {'logs': []})

    def adjust_logs(self, user: dict):
        if len(user['logs']) > self.max_logs:
            adjusted_logs = user['logs'][-self.max_logs:]
            self.update_user(user['name'], {'logs': adjusted_logs})
            return adjusted_logs
        return user['logs']

    # Job status: 1 = work, 0 = stop, -1 = kill
    def set_job_status(self, idx, status):
        if status == -1:
            self.delete_array_item(idx, 'Jobs')
        else:
            self.users.update_one({'Jobs.idx': idx}, {'$set': {'Jobs.$.status': status}})

    def get_job_status(self, idx):
        try:
            return self.users.find_one({'Jobs.idx': idx}, {'Jobs.$': 1})['Jobs'][0]['status']
        except TypeError:
            return -1
