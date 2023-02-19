from .backend import *

working_directory = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(working_directory, 'config.json'), 'r') as f:
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
