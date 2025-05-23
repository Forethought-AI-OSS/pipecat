#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import sys
import time
from typing import AsyncGenerator, List

from loguru import logger
from PIL import Image

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    MixerControlFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    SpriteFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import nanoseconds_to_seconds

BOT_VAD_STOP_SECS = 0.35


class BaseOutputTransport(FrameProcessor):
    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)

        self._params = params

        # Task to process incoming frames so we don't block upstream elements.
        self._sink_task = None

        # Task to process incoming frames using a clock.
        self._sink_clock_task = None

        # Task to write/send audio and image frames.
        self._video_out_task = None

        # These are the images that we should send at our desired framerate.
        self._video_images = None

        # Output sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0
        self._resampler = create_default_resampler()

        # Chunk size that will be written. It will be computed on StartFrame
        self._audio_chunk_size = 0
        self._audio_buffer = bytearray()

        self._stopped_event = asyncio.Event()

        # Indicates if the bot is currently speaking.
        self._bot_speaking = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self, frame: StartFrame):
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        # We will write 10ms*CHUNKS of audio at a time (where CHUNKS is the
        # `audio_out_10ms_chunks` parameter). If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * self._params.audio_out_10ms_chunks

        # Start audio mixer.
        if self._params.audio_out_mixer:
            await self._params.audio_out_mixer.start(self._sample_rate)
        self._create_video_task()
        self._create_sink_tasks()

    async def stop(self, frame: EndFrame):
        # Let the sink tasks process the queue until they reach this EndFrame.
        await self._sink_clock_queue.put((sys.maxsize, frame.id, frame))
        await self._sink_queue.put(frame)

        # At this point we have enqueued an EndFrame and we need to wait for
        # that EndFrame to be processed by the sink tasks. We also need to wait
        # for these tasks before cancelling the video and audio tasks below
        # because they might be still rendering.
        if self._sink_task:
            await self.wait_for_task(self._sink_task)
        if self._sink_clock_task:
            await self.wait_for_task(self._sink_clock_task)

        # We can now cancel the video task.
        await self._cancel_video_task()

    async def cancel(self, frame: CancelFrame):
        # Since we are cancelling everything it doesn't matter if we cancel sink
        # tasks first or not.
        await self._cancel_sink_tasks()
        await self._cancel_video_task()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        pass

    async def write_raw_video_frame(self, frame: OutputImageRawFrame):
        pass

    async def write_raw_audio_frames(self, frames: bytes):
        pass

    async def send_audio(self, frame: OutputAudioRawFrame):
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        #
        # System frames (like StartInterruptionFrame) are pushed
        # immediately. Other frames require order so they are put in the sink
        # queue.
        #
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (StartInterruptionFrame, StopInterruptionFrame)):
            await self.push_frame(frame, direction)
            await self._handle_interruptions(frame)
        elif isinstance(frame, TransportMessageUrgentFrame):
            await self.send_message(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames.
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            # Keep pushing EndFrame down so all the pipeline stops nicely.
            await self.push_frame(frame, direction)
        elif isinstance(frame, MixerControlFrame) and self._params.audio_out_mixer:
            await self._params.audio_out_mixer.process_frame(frame)
        # Other frames.
        elif isinstance(frame, OutputAudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, (OutputImageRawFrame, SpriteFrame)):
            await self._handle_image(frame)
        # TODO(aleix): Images and audio should support presentation timestamps.
        elif frame.pts:
            await self._sink_clock_queue.put((frame.pts, frame.id, frame))
        elif direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        else:
            await self._sink_queue.put(frame)

    async def _handle_interruptions(self, frame: Frame):
        if not self.interruptions_allowed:
            return

        if isinstance(frame, StartInterruptionFrame):
            # Cancel sink and video tasks.
            await self._cancel_sink_tasks()
            await self._cancel_video_task()
            # Create sink and video tasks.
            self._create_video_task()
            self._create_sink_tasks()
            # Let's send a bot stopped speaking if we have to.
            await self._bot_stopped_speaking()

    async def _handle_audio(self, frame: OutputAudioRawFrame):
        if not self._params.audio_out_enabled:
            return

        # We might need to resample if incoming audio doesn't match the
        # transport sample rate.
        resampled = await self._resampler.resample(
            frame.audio, frame.sample_rate, self._sample_rate
        )

        cls = type(frame)
        self._audio_buffer.extend(resampled)
        while len(self._audio_buffer) >= self._audio_chunk_size:
            chunk = cls(
                bytes(self._audio_buffer[: self._audio_chunk_size]),
                sample_rate=self._sample_rate,
                num_channels=frame.num_channels,
            )
            await self._sink_queue.put(chunk)
            self._audio_buffer = self._audio_buffer[self._audio_chunk_size :]

    async def _handle_image(self, frame: OutputImageRawFrame | SpriteFrame):
        if not self._params.video_out_enabled:
            return

        if self._params.video_out_is_live:
            await self._video_out_queue.put(frame)
        else:
            await self._sink_queue.put(frame)

    async def _bot_started_speaking(self):
        if not self._bot_speaking:
            logger.debug("Bot started speaking")
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = True

    async def _bot_stopped_speaking(self):
        if self._bot_speaking:
            logger.debug("Bot stopped speaking")
            await self.push_frame(BotStoppedSpeakingFrame())
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = False
            # Clean audio buffer (there could be tiny left overs if not multiple
            # to our output chunk size).
            self._audio_buffer = bytearray()

    #
    # Sink tasks
    #

    def _create_sink_tasks(self):
        if not self._sink_task:
            self._sink_queue = asyncio.Queue()
            self._sink_task = self.create_task(self._sink_task_handler())
        if not self._sink_clock_task:
            self._sink_clock_queue = asyncio.PriorityQueue()
            self._sink_clock_task = self.create_task(self._sink_clock_task_handler())

    async def _cancel_sink_tasks(self):
        # Stop sink tasks.
        if self._sink_task:
            await self.cancel_task(self._sink_task)
            self._sink_task = None
        # Stop sink clock tasks.
        if self._sink_clock_task:
            await self.cancel_task(self._sink_clock_task)
            self._sink_clock_task = None

    async def _sink_frame_handler(self, frame: Frame):
        if isinstance(frame, OutputImageRawFrame):
            await self._set_video_image(frame)
        elif isinstance(frame, SpriteFrame):
            await self._set_video_images(frame.images)
        elif isinstance(frame, TransportMessageFrame):
            await self.send_message(frame)

    async def _sink_clock_task_handler(self):
        running = True
        while running:
            try:
                timestamp, _, frame = await self._sink_clock_queue.get()

                # If we hit an EndFrame, we can finish right away.
                running = not isinstance(frame, EndFrame)

                # If we have a frame we check it's presentation timestamp. If it
                # has already passed we process it, otherwise we wait until it's
                # time to process it.
                if running:
                    current_time = self.get_clock().get_time()
                    if timestamp > current_time:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)

                    # Handle frame.
                    await self._sink_frame_handler(frame)

                    # Also, push frame downstream in case anyone else needs it.
                    await self.push_frame(frame)

                self._sink_clock_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"{self} error processing sink clock queue: {e}")

    def _next_frame(self) -> AsyncGenerator[Frame, None]:
        async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
            while True:
                try:
                    frame = await asyncio.wait_for(self._sink_queue.get(), timeout=vad_stop_secs)
                    yield frame
                except asyncio.TimeoutError:
                    # Notify the bot stopped speaking upstream if necessary.
                    await self._bot_stopped_speaking()

        async def with_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
            last_frame_time = 0
            silence = b"\x00" * self._audio_chunk_size
            while True:
                try:
                    frame = self._sink_queue.get_nowait()
                    if isinstance(frame, OutputAudioRawFrame):
                        frame.audio = await self._params.audio_out_mixer.mix(frame.audio)
                    last_frame_time = time.time()
                    yield frame
                except asyncio.QueueEmpty:
                    # Notify the bot stopped speaking upstream if necessary.
                    diff_time = time.time() - last_frame_time
                    if diff_time > vad_stop_secs:
                        await self._bot_stopped_speaking()
                    # Generate an audio frame with only the mixer's part.
                    frame = OutputAudioRawFrame(
                        audio=await self._params.audio_out_mixer.mix(silence),
                        sample_rate=self._sample_rate,
                        num_channels=self._params.audio_out_channels,
                    )
                    yield frame

        if self._params.audio_out_mixer:
            return with_mixer(BOT_VAD_STOP_SECS)
        else:
            return without_mixer(BOT_VAD_STOP_SECS)

    async def _sink_task_handler(self):
        # Push a BotSpeakingFrame every 200ms, we don't really need to push it
        # at every audio chunk. If the audio chunk is bigger than 200ms, push at
        # every audio chunk.
        TOTAL_CHUNK_MS = self._params.audio_out_10ms_chunks * 10
        BOT_SPEAKING_CHUNK_PERIOD = max(int(200 / TOTAL_CHUNK_MS), 1)
        bot_speaking_counter = 0
        async for frame in self._next_frame():
            # Notify the bot started speaking upstream if necessary and that
            # it's actually speaking.
            if isinstance(frame, TTSAudioRawFrame):
                await self._bot_started_speaking()
                if bot_speaking_counter % BOT_SPEAKING_CHUNK_PERIOD == 0:
                    await self.push_frame(BotSpeakingFrame())
                    await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
                    bot_speaking_counter = 0
                bot_speaking_counter += 1

            # No need to push EndFrame, it's pushed from process_frame().
            if isinstance(frame, EndFrame):
                break

            # Handle frame.
            await self._sink_frame_handler(frame)

            # Also, push frame downstream in case anyone else needs it.
            await self.push_frame(frame)

            # Send audio.
            if isinstance(frame, OutputAudioRawFrame):
                await self.write_raw_audio_frames(frame.audio)

    #
    # Video task
    #

    def _create_video_task(self):
        # Create video output queue and task if needed.
        if not self._video_out_task and self._params.video_out_enabled:
            self._video_out_queue = asyncio.Queue()
            self._video_out_task = self.create_task(self._video_out_task_handler())

    async def _cancel_video_task(self):
        # Stop video output task.
        if self._video_out_task and self._params.video_out_enabled:
            await self.cancel_task(self._video_out_task)
            self._video_out_task = None

    async def _draw_image(self, frame: OutputImageRawFrame):
        desired_size = (self._params.video_out_width, self._params.video_out_height)

        # TODO: we should refactor in the future to support dynamic resolutions
        # which is kind of what happens in P2P connections.
        # We need to add support for that inside the DailyTransport
        if frame.size != desired_size:
            image = Image.frombytes(frame.format, frame.size, frame.image)
            resized_image = image.resize(desired_size)
            # logger.warning(f"{frame} does not have the expected size {desired_size}, resizing")
            frame = OutputImageRawFrame(
                resized_image.tobytes(), resized_image.size, resized_image.format
            )

        await self.write_raw_video_frame(frame)

    async def _set_video_image(self, image: OutputImageRawFrame):
        self._video_images = itertools.cycle([image])

    async def _set_video_images(self, images: List[OutputImageRawFrame]):
        self._video_images = itertools.cycle(images)

    async def _video_out_task_handler(self):
        self._video_out_start_time = None
        self._video_out_frame_index = 0
        self._video_out_frame_duration = 1 / self._params.video_out_framerate
        self._video_out_frame_reset = self._video_out_frame_duration * 5
        while True:
            if self._params.video_out_is_live:
                await self._video_out_is_live_handler()
            elif self._video_images:
                image = next(self._video_images)
                await self._draw_image(image)
                await asyncio.sleep(self._video_out_frame_duration)
            else:
                await asyncio.sleep(self._video_out_frame_duration)

    async def _video_out_is_live_handler(self):
        image = await self._video_out_queue.get()

        # We get the start time as soon as we get the first image.
        if not self._video_out_start_time:
            self._video_out_start_time = time.time()
            self._video_out_frame_index = 0

        # Calculate how much time we need to wait before rendering next image.
        real_elapsed_time = time.time() - self._video_out_start_time
        real_render_time = self._video_out_frame_index * self._video_out_frame_duration
        delay_time = self._video_out_frame_duration + real_render_time - real_elapsed_time

        if abs(delay_time) > self._video_out_frame_reset:
            self._video_out_start_time = time.time()
            self._video_out_frame_index = 0
        elif delay_time > 0:
            await asyncio.sleep(delay_time)
            self._video_out_frame_index += 1

        # Render image
        await self._draw_image(image)

        self._video_out_queue.task_done()
