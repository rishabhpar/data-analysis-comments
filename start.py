from speech import AudioTranscripter
import time
import keyboard
import requests
import json


from pynput.keyboard import Key, Listener

sending_request ={
    "comment_text" : ""
}

audio = AudioTranscripter()
isRecording = False

def on_press(key):
    global audio, isRecording
    if key == Key.space and not isRecording:
        isRecording = True
        audio.start()
    #print('{0} pressed'.format(key))


def on_release(key):
    global audio, isRecording
    if key == Key.space and isRecording:
        isRecording = False
        text =audio.stop()
        sending_request["comment_text"]=text
        #print(sending_request)
        headers = {'Content-type': 'application/json'}
        response = requests.post(url="https://us-central1-comments-233718.cloudfunctions.net/predict_xgb",data=json.dumps(sending_request),headers=headers)
        print(response.text)

    #print('{0} release'.format(key))
    if key == Key.esc:
        # Stop listener
        return False


# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()