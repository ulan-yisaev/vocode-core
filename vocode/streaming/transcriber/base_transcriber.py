from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, TypeVar, Union

from vocode.streaming.models.audio import AudioEncoding
from vocode.streaming.models.transcriber import TranscriberConfig, Transcription
from vocode.streaming.utils.speed_manager import SpeedManager
from vocode.streaming.utils.worker import AbstractWorker, AsyncWorker, ThreadAsyncWorker

if TYPE_CHECKING:
    from vocode.streaming.streaming_conversation import StreamingConversation

TranscriberConfigType = TypeVar("TranscriberConfigType", bound=TranscriberConfig)


class AbstractTranscriber(Generic[TranscriberConfigType], AbstractWorker[bytes]):
    consumer: AbstractWorker[Transcription]
    streaming_conversation: "StreamingConversation"

    def __init__(self, transcriber_config: TranscriberConfigType):
        AbstractWorker.__init__(self)
        self.transcriber_config = transcriber_config
        self.is_muted = False
        self.speed_manager: Optional[SpeedManager] = None

    def attach_speed_manager(self, speed_manager: SpeedManager):
        self.speed_manager = speed_manager

    def mute(self):
        self.is_muted = True

    def unmute(self):
        self.is_muted = False

    def get_transcriber_config(self) -> TranscriberConfigType:
        return self.transcriber_config

    async def ready(self):
        return True

    def create_silent_chunk(self, chunk_size, sample_width=2):
        linear_audio = b"\0" * chunk_size

        if self.get_transcriber_config().audio_encoding == AudioEncoding.LINEAR16:
            return linear_audio
        elif self.get_transcriber_config().audio_encoding == AudioEncoding.MULAW:
            # Convert silent linear PCM to Mu-Law
            mu = 255
            max_val = 2**(8 * sample_width - 1) - 1  # Maximum value for given sample width

            def linear_to_mulaw(sample):
                # Constants for Mu-Law conversion
                sign = 0x80 if sample < 0 else 0
                magnitude = min(abs(sample), max_val)
                magnitude = (mu * np.log(1 + magnitude / max_val) / np.log(1 + mu))
                return (int(magnitude) | sign) ^ 0xFF

            # Generate an array of silent samples
            silent_samples = np.frombuffer(linear_audio, dtype=np.int16)
            ulaw_samples = bytearray(linear_to_mulaw(sample) for sample in silent_samples)
            return bytes(ulaw_samples)

    @abstractmethod
    async def _run_loop(self):
        pass

    def send_audio(self, chunk: bytes):
        if not self.is_muted:
            self.consume_nonblocking(chunk)
        else:
            self.consume_nonblocking(self.create_silent_chunk(len(chunk)))

    def produce_nonblocking(self, item: Transcription):
        self.consumer.consume_nonblocking(item)


class BaseAsyncTranscriber(AbstractTranscriber[TranscriberConfigType], AsyncWorker[bytes]):  # type: ignore
    def __init__(self, transcriber_config: TranscriberConfigType):
        AbstractTranscriber.__init__(self, transcriber_config)
        AsyncWorker.__init__(self)

    async def terminate(self):
        await AsyncWorker.terminate(self)


class BaseThreadAsyncTranscriber(  # type: ignore
    AbstractTranscriber[TranscriberConfigType], ThreadAsyncWorker[bytes]
):
    def __init__(self, transcriber_config: TranscriberConfigType):
        AbstractTranscriber.__init__(self, transcriber_config)
        ThreadAsyncWorker.__init__(self)

    def _run_loop(self):
        raise NotImplementedError

    async def run_thread_forwarding(self):
        try:
            await asyncio.gather(
                self._forward_to_thread(),
                self._forward_from_thread(),
            )
        except asyncio.CancelledError:
            return

    async def _forward_from_thread(self):
        while True:
            try:
                transcription = await self.output_janus_queue.async_q.get()
                self.consumer.consume_nonblocking(transcription)
            except asyncio.CancelledError:
                break

    def produce_nonblocking(self, item: Transcription):
        self.output_janus_queue.sync_q.put_nowait(item)

    async def terminate(self):
        await ThreadAsyncWorker.terminate(self)


BaseTranscriber = Union[
    BaseAsyncTranscriber[TranscriberConfigType],
    BaseThreadAsyncTranscriber[TranscriberConfigType],
]
