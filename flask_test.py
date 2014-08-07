from flask import Flask, render_template, request, jsonify, redirect
from flask.ext.socketio import SocketIO, emit, session
#from ptaco.libs import asr

app = Flask(__name__, static_url_path="/static", static_folder="trainweb/app/")
app.config['SERVER_NAME'] = "kronos:9999"
app.debug = True
#app.config.from_object({
#})
socketio = SocketIO(app)


@app.route('/')
def index():
    return redirect("/static/index.html")


@app.route('/recognize', methods=['POST'])
def recognize():
    response = recognize_wav(request.data)

    return jsonify(response)


@socketio.on('begin')
def begin_recognition(message):
    session['recognizer'] = OnlineASR(emit)


@socketio.on('chunk')
def recognize_chunk(message):
    session['recognizer'].recognize_chunk(message)


@socketio.on('end')
def end_recognition(message):
    session['recognizer'].end()


if __name__ == '__main__':
    app.secret_key = 12345
    print "Launching"
    socketio.run(app, host="0.0.0.0")