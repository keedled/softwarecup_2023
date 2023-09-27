from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime
db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nickname = db.Column(db.String(12), unique=True)
    username = db.Column(db.String(12), unique=True)
    password = db.Column(db.String(128))
    role = db.Column(db.Integer,default=1)
    is_banned = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'nickname': self.nickname,
            'username': self.username,
            'role': self.role,
            'is_banned': self.is_banned,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    dataset_name = db.Column(db.String(255))
    score = db.Column(db.Float)
    file_path = db.Column(db.String(255))
    total_time=db.Column(db.Float)
    url1 = db.Column(db.String(500))
    url2 = db.Column(db.String(500))
    url3 = db.Column(db.String(500))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref=db.backref('models', lazy=True))
    created_at = db.Column(db.DateTime, default=datetime.now)
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_image_url = db.Column(db.String(500))
    result_json_url = db.Column(db.String(500))
    class0_count = db.Column(db.Integer)
    class1_count = db.Column(db.Integer)
    class2_count = db.Column(db.Integer)
    class3_count = db.Column(db.Integer)
    class4_count = db.Column(db.Integer)
    class5_count = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))
    created_at = db.Column(db.DateTime, default=datetime.now)
class Announcement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class UserLogin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'))
    user = relationship('User', backref=db.backref('logins', lazy=True))
    login_time = db.Column(db.DateTime, default=datetime.now)
def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
