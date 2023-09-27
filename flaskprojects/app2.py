from flask import request, render_template, current_app, send_file, jsonify, abort, send_from_directory
from werkzeug.utils import secure_filename
from database import db, User, Model, Prediction,Announcement,UserLogin
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from logging_module import setup_logger
from task import train_model, test_model, app
from datetime import datetime, timedelta
from waitress import serve
from sqlalchemy.sql import func
import os


# 日志部分
class LoggingMiddleware(object):
    def __init__(self, app, logger):
        self.app = app
        self.logger = logger

    def __call__(self, environ, start_response):
        self.logger.info('Request details: Method - %s, Path - %s', environ['REQUEST_METHOD'], environ['PATH_INFO'])
        return self.app(environ, start_response)


# 日志部
logger = setup_logger()
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data['username']
        password = data['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            if user.is_banned == 0:
                return jsonify({'status': 'banned', 'message': '该账号已被封禁'})
            login_user(user)
            login_record = UserLogin(user_id=user.id)
            db.session.add(login_record)
            db.session.commit()
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': '账号或密码错误'})
    return render_template('login.html')


@app.route('/train_img/<path:filename>')
def serve_image(filename):
    return send_from_directory('train_img', filename)


@app.route('/results/<path:filename>')
def custom_static(filename):
    return send_from_directory('results', filename)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'status': 'success'})


@app.route('/get_nickname', methods=['GET'])
@login_required
def get_nickname():
    return jsonify({'nickname': current_user.nickname,'role': current_user.role})
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            nickname = request.form['nickname']
            username = request.form['username']
            password = request.form['password']
            if User.query.filter_by(username=username).first():
                return jsonify(status="error", message="该用户名已被注册"), 400
            if User.query.filter_by(nickname=nickname).first():
                return jsonify(status="error", message="该昵称已被注册"), 400
            user = User(nickname=nickname, username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            return jsonify(status="success", message="注册成功")
        except Exception as e:
            print(e)
            return jsonify(status="error", message="注册失败"), 500
    else:
        return render_template('register.html')


@app.route('/nr', methods=['GET', 'POST'])
def nr():
    if request.method == 'POST':
        # return redirect(url_for('login'))
        pass
    return render_template('nr.html')
@app.route('/fh',methods=['GET', 'POST'])
def fh():
    return render_template('fh.html')

@app.route('/count_m_p', methods=['GET'])
def count_m_p():
    user_id = current_user.id
    model_count = Model.query.filter_by(user_id=user_id).count()
    prediction_count = Prediction.query.filter_by(user_id=user_id).count()
    return jsonify({
        'model_count': model_count,
        'prediction_count': prediction_count
    })
@app.route('/avg_count', methods=['GET'])
@login_required
def avg_count():
    predictions = Prediction.query.filter_by(user_id=current_user.id)
    averages = {}
    for i in range(6):
        class_count = getattr(Prediction, f'class{i}_count')
        avg = db.session.query(func.avg(class_count)).filter_by(user_id=current_user.id).scalar()
        averages[f'class{i}_count'] = round(avg, 2) if avg else 0
    return jsonify(averages)
@app.route('/models', methods=['GET'])
@login_required
def get_models():
    user_id = current_user.id
    models = Model.query.filter_by(user_id=user_id).all()
    models_list = []
    for model in models:
        model_dict = {
            'id': model.id,
            'name': model.name,
            'file_path': model.file_path,
            'user_id': model.user_id,
            'dataset_name': model.dataset_name,
            'score': model.score,
            'time': model.total_time,
            'url1': model.url1,
            'url2': model.url2,
            'url3': model.url3
        }
        models_list.append(model_dict)
    return jsonify(models_list)


@app.route('/models2', methods=['GET'])
@login_required
def get_models2():
    user_id = current_user.id
    models = Model.query.filter_by(user_id=user_id)
    models_list = []
    user_friendly_id = 1
    for model in models:
        model_dict = {
            'id': model.id,
            'userFriendlyId': user_friendly_id,
            'name': model.name,
        }
        models_list.append(model_dict)
        user_friendly_id += 1
    return jsonify(models_list)


@app.route('/download/<int:id>')
def download(id):
    model = Model.query.get(id)
    if model is None:
        abort(404)
    base_dir = current_app.config['BASE_DIR']
    file_path = os.path.join(base_dir, model.file_path)
    print(file_path)
    return send_file(file_path)


@app.route('/delete/<int:id>', methods=['DELETE'])
def delete(id):
    model = Model.query.get(id)
    model_name = model.name
    try:
        model_name_with_id = f"{model.id}_{model_name}"
        os.remove(f"models/{model_name_with_id}.pkl")
    except FileNotFoundError:
        pass
    try:
        os.remove(f"train_img/total/img1_{model.id}.png")
    except FileNotFoundError:
        pass
    try:
        os.remove(f"train_img/accuracy/img2_{model.id}.png")
    except FileNotFoundError:
        pass
    try:
        os.remove(f"train_img/mult/img3_{model.id}.png")
    except FileNotFoundError:
        pass
    db.session.delete(model)
    db.session.commit()
    return {'success': True}


@app.route('/edit', methods=['POST'])
def edit():
    data = request.get_json()
    id = data['id']
    name = data['name']
    model = Model.query.get(id)
    if model is not None:
        old_name = model.name
        old_file_path = f'models/{old_name}.pkl'
        new_file_path = f'models/{name}.pkl'
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
        model.name = name
        model.file_path = new_file_path
        db.session.commit()
    if model is None:
        return {'success': False, 'message': '模型不存在'}, 400
    return {'success': True}


@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']
    model_name = request.form['modelName']
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': '训练集未上传'}), 400
    if 'modelName' not in request.form:
        return jsonify({'status': 'error', 'message': '没有输入模型名字'}), 400
    user_id = current_user.id
    filename = secure_filename(file.filename)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_folder = os.path.join(base_dir, 'datasets')
    file_path = os.path.join(upload_folder, filename)
    print(file_path)
    if not os.path.isfile(file_path):
        file.save(file_path)
    task = train_model.apply_async(args=[file_path, model_name, user_id, file.filename])
    return jsonify({'task_id': task.id, 'state': task.state}), 202
@app.route('/train_progress/<task_id>', methods=['GET'])
def get_train_progress(task_id):
    progress = app.redis_conn.get(f"train_progress_{task_id}")
    if progress:
        return jsonify({"progress": float(progress)})
    else:
        return jsonify({"progress": None}), 404

@app.route('/test', methods=['POST'])
def test():
    file = request.files['test_data']
    model_id = request.form['model_id']
    if 'test_data' not in request.files:
        return jsonify({'status': 'error', 'message': '没有验证集上传'}), 400
    if 'model_id' not in request.form:
        return jsonify({'status': 'error', 'message': '无模型'}), 400
    user_id = current_user.id
    filename = secure_filename(file.filename)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    upload_folder = os.path.join(base_dir, 'datasets')
    file_path = os.path.join(upload_folder, filename)
    print(file_path)
    if not os.path.isfile(file_path):
        file.save(file_path)
    task = test_model.apply_async(args=[file_path, model_id, user_id])
    return jsonify({'task_id': task.id, 'state': task.state}), 202

@app.route('/predictions', methods=['GET'])
@login_required
def get_predictions():
    user_id = current_user.id
    predictions = Prediction.query.filter_by(user_id=user_id).all()
    predictions_list = []
    for prediction in predictions:
        prediction_dict = {
            'id': prediction.id,
            'result_image_url': prediction.result_image_url,
            'class0_count': prediction.class0_count,
            'class1_count': prediction.class1_count,
            'class2_count': prediction.class2_count,
            'class3_count': prediction.class3_count,
            'class4_count': prediction.class4_count,
            'class5_count': prediction.class5_count,
        }
        predictions_list.append(prediction_dict)
    return jsonify(predictions_list)
@app.route('/delete_result/<int:id>', methods=['DELETE'])
def delete_result(id):
    prediction = Prediction.query.get(id)
    try:
        os.remove(f"results/img/result_{id}.png")
    except FileNotFoundError:
        pass
    try:
        os.remove(f"results/json/{id}.json")
    except FileNotFoundError:
        pass
    db.session.delete(prediction)
    db.session.commit()
    return {'success': True}

@app.route('/download_result/<int:id>')
def download_result(id):
    prediction = Prediction.query.get(id)
    if prediction is None:
        abort(404)
    base_dir = current_app.config['BASE_DIR']
    file_path = os.path.join(base_dir, prediction.result_json_url)
    print(file_path)
    return send_file(file_path, as_attachment=True)
@app.route('/five_day_data',methods=['GET'])
@login_required
def five_day_data():
    total_users = User.query.count()
    total_models = Model.query.count()
    total_predictions = Prediction.query.count()
    daily_data = {}
    for days_ago in range(4, -1, -1):
        date = (datetime.now() - timedelta(days=days_ago)).date()
        new_users = db.session.query(User).filter(func.date(User.created_at) == date).count()
        new_models = db.session.query(Model).filter(func.date(Model.created_at) == date).count()
        new_predictions = db.session.query(Prediction).filter(func.date(Prediction.created_at) == date).count()

        daily_data[str(date)] = {
            'new_users': new_users,
            'new_models': new_models,
            'new_predictions': new_predictions,
        }
    data = {
        'total_users': total_users,
        'total_models': total_models,
        'total_predictions': total_predictions,
        'daily_data': daily_data
    }
    return jsonify(data)
def has_permission(operator_role, target_role):
    if operator_role == 3 and target_role in [1, 2]:
        return True
    elif operator_role == 2 and target_role == 1:
        return True
    return False

@app.route('/ban/<int:user_id>', methods=['POST'])
def ban_user(user_id):
    operator = current_user
    target_user = User.query.get_or_404(user_id)  # 获取被操作的用户
    print(1)
    print(target_user)
    if has_permission(operator.role, target_user.role):
        target_user.is_banned = 0
        db.session.commit()
        return jsonify({"message": "用户已被封号"})
    else:
        return jsonify({"message": "无权限操作"}), 403

@app.route('/unban/<int:user_id>', methods=['POST'])
def unban_user(user_id):
    operator = current_user
    target_user = User.query.get_or_404(user_id)  # 获取被操作的用户

    if has_permission(operator.role, target_user.role):
        target_user.is_banned = 1
        db.session.commit()
        return jsonify({"message": "用户已解除封号"})
    else:
        return jsonify({"message": "无权限操作"}), 403

@app.route('/users', methods=['GET'])
def get_all_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])
@app.route('/announcement', methods=['POST'])
def update_or_create_announcement():
    data = request.json
    content = data.get('content')

    if not content:
        return jsonify({"error": "Content is required!"}), 400

    # Fetch the existing announcement, if it exists
    announcement = Announcement.query.first()

    if announcement:
        # Update the existing announcement
        announcement.content = content
    else:
        # Create a new announcement
        announcement = Announcement(content=content)
        db.session.add(announcement)

    db.session.commit()
    return jsonify({"message": "Announcement updated successfully"}), 200

@app.route('/announcement', methods=['GET'])
def get_announcement():
    announcement = Announcement.query.first()
    content = announcement.content if announcement else "No announcement available"
    return jsonify({"content": content}), 200
@app.route('/get_user_details/<int:user_id>', methods=['GET'])
def get_user_details(user_id):

  model = Model.query.filter_by(user_id=user_id).order_by(Model.created_at.desc()).first()
  model_latest_time = model.created_at if model else None

  prediction = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).first()
  prediction_latest_time = prediction.created_at if prediction else None

  login = UserLogin.query.filter_by(user_id=user_id).order_by(UserLogin.login_time.desc()).first()
  login_latest_time = login.login_time if login else None

  model_count = Model.query.filter_by(user_id=user_id).count()
  prediction_count = Prediction.query.filter_by(user_id=user_id).count()
  print(model_latest_time)
  print(login_latest_time)
  print(prediction_latest_time)
  return jsonify({
    'model_count': model_count,
    'prediction_count': prediction_count,
    'model_latest_time': model_latest_time,
    'prediction_latest_time': prediction_latest_time,
    'login_latest_time': login_latest_time
  })
@app.route('/set-admin', methods=['POST'])
def set_admin():
    user_id = request.json.get('userId')
    user = User.query.get(user_id)
    if user:
        user.role = 2
        db.session.commit()
        return jsonify(success=True), 200
    else:
        return jsonify(success=False, message="User not found"), 400

@app.route('/remove-admin', methods=['POST'])
def remove_admin():
    user_id = request.json.get('userId')
    user = User.query.get(user_id)
    if user:
        user.role = 1
        db.session.commit()
        return jsonify(success=True), 200
    else:
        return jsonify(success=False, message="User not found"), 400


@app.route('/login_data/<int:user_id>', methods=['GET'])
def get_login_data(user_id):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=6)
    logins = UserLogin.query.filter(UserLogin.user_id == user_id,
                                    UserLogin.login_time.between(start_date, end_date)).all()
    data = {}
    for i in range(7):
        date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
        data[date] = 0
    for login in logins:
        date = login.login_time.strftime('%Y-%m-%d')
        data[date] += 1
    return jsonify(data)


@app.route('/login_data_detail/<int:user_id>/<date>', methods=['GET'])
def get_login_data_detail(user_id, date):
    print(date)
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Expected YYYY-MM-DD."}), 400
    print(date_obj)
    logins = UserLogin.query.filter(
        UserLogin.user_id == user_id,
        UserLogin.login_time.between(date_obj, date_obj + timedelta(days=1))
    ).all()

    intervals = {
        "00:00-03:00": 0,
        "03:00-06:00": 0,
        "06:00-09:00": 0,
        "09:00-12:00": 0,
        "12:00-15:00": 0,
        "15:00-18:00": 0,
        "18:00-21:00": 0,
        "21:00-24:00": 0,
    }

    for login in logins:
        hour = login.login_time.hour
        if 0 <= hour < 3:
            intervals["00:00-03:00"] += 1
        elif 3 <= hour < 6:
            intervals["03:00-06:00"] += 1
        elif 6 <= hour < 9:
            intervals["06:00-09:00"] += 1
        elif 9 <= hour < 12:
            intervals["09:00-12:00"] += 1
        elif 12 <= hour < 15:
            intervals["12:00-15:00"] += 1
        elif 15 <= hour < 18:
            intervals["15:00-18:00"] += 1
        elif 18 <= hour < 21:
            intervals["18:00-21:00"] += 1
        else:
            intervals["21:00-24:00"] += 1

    return jsonify(intervals)


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5002)
