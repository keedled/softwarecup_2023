@echo off

set command1=python -m waitress --listen=localhost:5000 app:app
set command2=python -m waitress --listen=localhost:5001 app1:app
set command3=python -m waitress --listen=localhost:5002 app2:app
set command4=celery -A task.celery worker  -l info -P eventlet

start powershell -noexit -command %command1%
start powershell -noexit -command %command2%  
start powershell -noexit -command %command3%
start powershell -noexit -command %command4%