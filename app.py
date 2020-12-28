import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
from VoiceAuthentication import VoiceAuthentication
import app_setup


application = Flask(__name__)

VOICE_AUTH_MODEL = VoiceAuthentication('mobile_model.pt')


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/verify', methods=['POST'])
def verify():
    '''
    verify from wav files
    '''
    def get_and_save_temp_file(request_file):
        file_storage = request.files[request_file]
        temp_path = os.path.join(app_setup.TEMP_FOLDER, file_storage.filename)
        file_storage.save(temp_path)
        return temp_path

    f1_path = get_and_save_temp_file("wav1")
    f2_path = get_and_save_temp_file("wav2")

    result = VOICE_AUTH_MODEL.authenticate(f1_path, f2_path)

    # return render_template('index.html', display=result[0])
    return jsonify(verification=result[0],
                   votes=result[1],
                   percentage="{}%".format(result[2]))

@application.route('/register', methods=['POST'])
def register():
    '''
    verify from wav files
    '''
    def register_wav(wav_file):
        file_storage = request.files[wav_file]
        saved_path = os.path.join(app_setup.TEMP_FOLDER, file_storage.filename)
        file_storage.save(saved_path)
        return saved_path

    f1_path = register_wav("wav1")       
    result = VOICE_AUTH_MODEL.register(f1_path)
    if result == "null":
        return jsonify(Registration_status="Unsuccessful: Cannot register null audio file. Please use a valid audio of atleast 5 second long.")
    elif result == False:
        return jsonify(name="File already Exists",
                       status="Registeration unsuccessful")
    else:
        return jsonify(name=f1_path,
                       status="Registered Successfully")

@application.route('/recognize', methods=['POST'])
def recognize():
    '''
    verify from wav files
    '''
    def _recognize(wav_file):
        file_storage = request.files[wav_file]
        saved_path = os.path.join(app_setup.TEST_FOLDER, file_storage.filename)
        file_storage.save(saved_path)
        return saved_path
    f1_path = _recognize("wav1")

    results = VOICE_AUTH_MODEL.recognize(f1_path)
    if results == "null":
        return jsonify(Registration_status="Unsuccessful: Cannot register null audio file. Please use a valid audio of atleast 5 second long.")
    results_all = results[4]
      
    return jsonify(results_all) 

if __name__ == "__main__":
    application.run(debug=True)
