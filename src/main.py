from pathlib import Path
from dotenv import load_dotenv
import openai
import wave
import sys
import pyaudio
import audioop
import queue
import tempfile
import threading
import os

load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 5
THRESHOLD = 160

def stt(raw_audio_chunks, sample_size):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file_path = Path(temp_dir).resolve() / 'audio.wav'

        with wave.open(str(audio_file_path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_size)
            wf.setframerate(RATE)

            for chunk in raw_audio_chunks:
                wf.writeframes(chunk)

        try:
            audio_file= open(audio_file_path, "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            audio_file.close()
            os.remove(audio_file_path)

            return transcript
        except Exception as e:
            print(repr(e))
            return None


def transcribe_loop(audio_queue, sample_size):
    while True:
        audio_data = audio_queue.get()
        print(stt(audio_data, sample_size))


def main():
    audio_queue = queue.Queue()
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
    sample_size = p.get_sample_size(FORMAT)

    threading.Thread(target=transcribe_loop, args=(audio_queue, sample_size)).start()

    audios = []
    recording = False
    stop_counter = 30
    chunks = []
    while True:
        chunk = stream.read(CHUNK)
        rms = audioop.rms(chunk, CHANNELS)
        has_audio = rms >= THRESHOLD

        if has_audio:
            recording = True
            stop_counter = 30
        else:
            stop_counter -= 1

        if stop_counter == 0:
            audio_queue.put_nowait(chunks)
            recording = False
            chunks = []

        if recording:
            chunks.append(chunk)

    stream.close()
    p.terminate()

main()