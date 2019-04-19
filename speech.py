from pynput import keyboard
import pyaudio
import time
import wave
import speech_recognition as sr
import base64
import requests
import json
CHUNK = 8192
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = 'output.wav'
AMBIENT_FILENAME = 'ambient.wav'
API_KEY = "AIzaSyDxPsx9KoRmeK5AnGczYulBC-Qf--RTKLE"

p = pyaudio.PyAudio()
frames = []


class AudioTranscripter:
    def __init__(self):
        self.listener = MyListener()
        self.id = None
        #self.google = GoogleTranscripter()
        self.listener.start()
        self.listener.stream.start_stream()
        self.start()
        time.sleep(1.5)
        self.stop(AMBIENT_FILENAME)
        file = sr.AudioFile(AMBIENT_FILENAME)
        #with file as source:
            #self.google.r.adjust_for_ambient_noise(source)
        self.text_ready = False

    def start(self):
        self.listener.key_pressed = True
        frames.clear()

    def stop(self, filename=WAVE_OUTPUT_FILENAME):
        self.listener.key_pressed = None
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        if filename != AMBIENT_FILENAME:
            try:
                google_text = getText(filename)
                print(google_text)
                #self.id = resp['id']
                return google_text
            except Exception:
                print("Couldn't Recognize")
                return None
        return None


class MyListener(keyboard.Listener):

    def __init__(self):
        super(MyListener, self).__init__(self.on_press, self.on_release)
        self.key_pressed = None

        self.stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback,
        )

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.key_pressed = True

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.key_pressed = False

    def callback(
            self,
            in_data,
            frame_count,
            time_info,
            status,
    ):
        if self.key_pressed:
            frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paContinue)

sending_request ={
    "config": {
    	"languageCode": "en-US",
    	"maxAlternatives": 1,
    	"profanityFilter": False
	},
    "audio": {
        "content": ""
    }
}




def getText(filename):
    global API_KEY
    file = open(filename, "rb")
    file=file.read()
    out = open("testfile.txt", "w")
    out.write(base64.urlsafe_b64encode(file).decode("ascii"))

    #audio = self.r.record(source)
    sending_request["audio"]["content"] = base64.urlsafe_b64encode(file).decode("ascii")
    try:
        response = requests.post(
            url='https://speech.googleapis.com/v1/speech:recognize?key={}'.format(API_KEY),
            # import json module
            # dumps the object to JSON
            data=json.dumps(sending_request),
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    #json_data = json.loads(response)
    return json.loads(response.text)['results'][0]['alternatives'][0]['transcript']
    #return json_data['alternatives'][0]['transcript']
