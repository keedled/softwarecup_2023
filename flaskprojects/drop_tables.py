from sqlalchemy import inspect
from app import db, app
from database import User,Announcement,UserLogin
def print_all_tables():
    with app.app_context():
        inspector = inspect(db.engine)
        for table_name in inspector.get_table_names():
            print(f"Table: {table_name}")

def drop_all_tables():
    with app.app_context():
        db.drop_all()

def print_users_table():
    with app.app_context():
        users = User.query.all()
        for user in users:
            print(f"ID: {user.id}, Nickname: {user.nickname}, Username: {user.username}, Password: {user.password}, Role: {user.role}, Created at: {user.created_at},is_banned:{user.is_banned}")
def print_announcements_table():
    with app.app_context():
        announcements = Announcement.query.all()
        for announcement in announcements:
            print(f"ID: {announcement.id}, Content: {announcement.content}, Timestamp: {announcement.timestamp}")

def change_user_role(user_id, new_role):
    with app.app_context():
        user = User.query.get(user_id)
        if user:
            user.role = new_role
            db.session.commit()
            print(f"User ID: {user.id}, Role changed to: {user.role}")
        else:
            print(f"No user found with ID: {user_id}")

def print_userlogin():
    with app.app_context():
        userlogins = UserLogin.query.all()
        for userlogin in userlogins:
            print(f"ID:{userlogin.id}, User_Id:{userlogin.user_id}, Time:{userlogin.login_time}")

if __name__ == "__main__":
    print_all_tables()
    change_user_role(1, 3)
    print_users_table()
    print_announcements_table()
    print_userlogin()
    # drop_all_tables()
    print("All tables dropped.")
