from flask import Flask
from celery import Celery
from flask_session import Session
from database import init_db
from flask_cors import CORS
import redis
import logging
def create_app():
    app = Flask(__name__)
    CORS(app)
    # app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:qixinkai123@localhost/db_01'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
    app.config['broker_url'] = 'redis://127.0.0.1:6379/0'
    app.config['result_backend'] = 'redis://127.0.0.1:6379/0'
    app.redis_conn = redis.StrictRedis(host='127.0.0.1', port=6379, db=1)
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['SESSION_KEY_PREFIX'] = 'session'
    app.config['SESSION_REDIS'] = redis.StrictRedis(host='127.0.0.1', port=6379, db=2)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['BASE_DIR'] = 'E:/flaskprojects/'
    app.secret_key = 'qixinkai1'
    init_db(app)
    Session(app)
    return app
def make_celery(app):
    celery = Celery(app.import_name, broker=app.config['broker_url'], backend=app.config['result_backend'])
    celery.conf.update(app.config)
    celery_logger = logging.getLogger('celery')
    celery_logger.setLevel(logging.INFO)
    celery_logger.addHandler(logging.FileHandler('celery.log'))
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
        @property
        def redis(self):
            return app.redis_conn

    celery.Task = ContextTask
    celery.conf.update(
        broker_connection_retry_on_startup=True
    )
    return celery