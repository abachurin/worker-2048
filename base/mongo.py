from pymongo import MongoClient


class Mongo:

    max_logs = 500
    FIELDS = ('Agents', 'Games', 'Jobs')

    def __init__(self, credentials: dict):
        self.cluster = f'mongodb+srv://{credentials["user"]}:{credentials["pwd"]}@instance-0' \
                       f'.55byx.mongodb.net/?retryWrites=true&w=majority'
        self.client = MongoClient(self.cluster)
        self.db = self.client['robot-2048']
        self.users = self.db['users']
        self.array_names = ('Agents', 'Games', 'Jobs')

    def find_user(self, name: str):
        user = self.users.find_one({'name': name})
        if user is not None:
            del user['_id']
        return user

    def delete_user(self, name: str):
        self.users.delete_one({'name': name})

    def new_user(self, name: str, pwd: str, status: str):
        user = {
            'name': name,
            'pwd': pwd,
            'status': status,
            'Agents': [
                # {
                #     'idx': 'A',
                #     'n': 4,
                #     'alpha': 0.2,
                #     'decay': 0.9,
                #     'step': 2000,
                #     'min_alpha': 0.01
                # }
            ],
            'Games': [
                # {
                #     'idx': 'Best_of_A',
                #     'initial': [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                #     'moves': [0, 1, 2, 3],
                #     'tiles': [[0, 3, 2], [3, 0, 1], [2, 3, 1], [2, 1, 1]]
                # }
            ],
            'Jobs': [
                # {
                #     'idx': 1,
                #     'status': 2
                # },
                # {
                #     'idx': 2,
                #     'status': 3
                # },
                # {
                #     'idx': 3,
                #     'status': 5
                # }
            ],
            'logs': [f'Hello {name}! Click Help if unsure what this is about']
        }
        self.users.insert_one(user)
        user.pop('_id')
        return user

    def update_user(self, name: str, fields: dict):
        self.users.update_one({'name': name}, {'$set': fields})

    def all_items(self, kind: str):
        if kind in self.array_names:
            kind += '.idx'
        return self.users.distinct(kind)

    def admin_list(self):
        return [user['name'] for user in self.users.find({'status': 'admin'})]

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
