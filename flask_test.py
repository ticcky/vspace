from flask import Flask, render_template
from flask.ext.socketio import SocketIO, emit

app = Flask(__name__) #, static_url_path="/static", static_folder="trainweb/app/")
app.debug = True
app.secret_key = 'secret!'
app.config['SERVER_NAME'] = 'kronos:9999'
socketio = SocketIO(app)

#@app.route('/')
#def index():
#    return render_template('index.html')

#@socketio.on('my event')
#def test_message(message):
#    emit('my response', {'data': 'got it!'})

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0")