import queue
import threading
from abc import ABC, abstractmethod

import pyaudio
from piper.voice import PiperVoice
from typing_extensions import override


class AudioPlayerBase(ABC):
    @abstractmethod
    def feed(self, text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_until_done(self) -> None:
        raise NotImplementedError


class AudioPlayer(threading.Thread, AudioPlayerBase):
    def __init__(self, voice: PiperVoice):
        super().__init__()
        self.voice: PiperVoice = voice
        self.sample_rate: int = voice.config.sample_rate
        self.queue: queue.Queue[bytes | None] = queue.Queue()

    @override
    def run(self):
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True
            )
            while True:
                chunk = self.queue.get()
                if chunk is None:
                    self.queue.task_done()
                    break
                stream.write(chunk)
        finally:
            if stream:
                stream.close()
            p.terminate()

    @override
    def feed(self, text: str):
        for chunk in self.voice.synthesize(text):
            self.queue.put_nowait(chunk.audio_int16_bytes)

    @override
    def stop(self):
        self.queue.put_nowait(None)

    @override
    def wait_until_done(self):
        self.join()


class NoAudio(AudioPlayerBase):
    @override
    def feed(self, text: str):
        pass

    @override
    def stop(self):
        pass

    @override
    def wait_until_done(self):
        pass


class TTS:
    def __init__(self, model: str):
        self.model: str = model.strip()

    def create_audio(self) -> AudioPlayerBase:
        if self.model:
            voice = PiperVoice.load(self.model)
            audio_player = AudioPlayer(voice=voice)
            audio_player.start()
            return audio_player
        else:
            return NoAudio()
