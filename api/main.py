import logging
import os
import time
import uuid
from multiprocessing import Pool

import requests

from modal import (
    Dict,
    Image,
    Mount,
    Period,
    Secret,
    SharedVolume,
    Stub,
    method,
    asgi_app,
)

volume = SharedVolume().persist("whisper-cache-volume")

api_image = Image.from_dockerhub(
    # CUDA and CUDNN required to install JAX
    "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
    setup_dockerfile_commands=[
        "RUN apt-get update",
        "RUN apt-get install -y python3 python3-pip python-is-python3",
    ],
).pip_install("https://github.com/sanchit-gandhi/whisper-jax", "requests")

stub = Stub(
    "speedy-whisper-api",
    image=api_image,
)

stub.running_jobs = Dict()
## LOOK AT THIS SHIT - WHAT IS CC
cc.initialize_cache("./jax_cache")  # TODO: Move to Modal SharedVolume
checkpoint = "openai/whisper-large-v2"

BATCH_SIZE = 32
CHUNK_LENGTH_S = 30
NUM_PROC = 32
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 7200  # limit to 2 hour YouTube files

language_names = sorted(TO_LANGUAGE_CODE.keys())
## LANGUAGE NAMES?? FUCK!!!


logger = logging.getLogger("speedy-whisper-api")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def identity(batch):
    return batch


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


@stub.cls(cpu=4)
class Whisper:
    # Code below executes on container start
    # https://modal.com/docs/guide/lifecycle-functions#__enter__-and-__aenter__
    def __enter__(self):
        self.pipeline = FlaxWhisperPipline(
            checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE
        )
        stride_length_s = CHUNK_LENGTH_S / 6
        chunk_len = round(
            CHUNK_LENGTH_S * self.pipeline.feature_extractor.sampling_rate
        )
        stride_left = stride_right = round(
            stride_length_s * self.pipeline.feature_extractor.sampling_rate
        )
        step = chunk_len - stride_left - stride_right
        pool = Pool(NUM_PROC)

        # Pre-compile Whisper-JAX code to container architecture. Subsequent function
        # calls will call this cached/compiled code which will be super-fast.
        logger.info("Compiling forward call...")
        start = time.time()
        random_inputs = {"input_features": np.ones((BATCH_SIZE, 80, 3000))}
        random_timestamps = pipeline.forward(
            random_inputs, batch_size=BATCH_SIZE, return_timestamps=True
        )
        compile_time = time.time() - start
        logger.info(f"Compiled in {compile_time}s")

    @method()
    def transcribe(audio_url, task, return_timestamps):
        job_id = str(uuid.uuid4())
        logger.info(f"Starting job {job_id}")

        logger.info(f"{job_id}: Downloading file...")
        response = requests.get(audio_url)
        if response.status_code != 200:
            logger.warning("Failed to download audio file")
            raise ValueError(
                "Failed to download audio file from URL. Please check the URL and try again."
            )
        logger.info(f"{job_id}: Download complete.")
        file = response.content

        # Chunk file using ffmpeg
        logger.info(f"{job_id}: Chunking file...")

        try:
            inputs = ffmpeg_read(file, pipeline.feature_extractor.sampling_rate)
        except Exception:
            logger.warning(f"{job_id}: Failed to chunk audio file.")
            raise ValueError(
                "Failed to chunk audio file using ffmpeg. Please ensure the file is a valid audio file and try again."
            )
        inputs = {
            "array": inputs,
            "sampling_rate": pipeline.feature_extractor.sampling_rate,
        }
        logger.info(f"{job_id}: Chunked audio file.")
        logger.info(f"{job_id}: Transcribing using Whisper-JAX...")
        text, runtime = tqdm_generate(
            inputs, task=task, return_timestamps=return_timestamps
        )
        logger.info(f"{job_id}: Transcription complete.")
        return text, runtime


###

import jax.numpy as jnp
import numpy as np
from jax.experimental.compilation_cache import compilation_cache as cc
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers.pipelines.audio_utils import ffmpeg_read

from whisper_jax import FlaxWhisperPipline

cc.initialize_cache("./jax_cache")  # TODO: Move to Modal SharedVolume
checkpoint = "openai/whisper-large-v2"

BATCH_SIZE = 32
CHUNK_LENGTH_S = 30
NUM_PROC = 32
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 7200  # limit to 2 hour YouTube files

language_names = sorted(TO_LANGUAGE_CODE.keys())

logger = logging.getLogger("whisper-jax-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def identity(batch):
    return batch


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


if __name__ == "__main__":
    pipeline = FlaxWhisperPipline(
        checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE
    )  # Can be optimised using 16 bit floating point math
    stride_length_s = CHUNK_LENGTH_S / 6
    chunk_len = round(CHUNK_LENGTH_S * pipeline.feature_extractor.sampling_rate)
    stride_left = stride_right = round(
        stride_length_s * pipeline.feature_extractor.sampling_rate
    )
    step = chunk_len - stride_left - stride_right
    pool = Pool(NUM_PROC)

    # do a pre-compile step so that the first user to use the demo isn't hit with a long transcription time
    logger.info("compiling forward call...")
    start = time.time()
    random_inputs = {"input_features": np.ones((BATCH_SIZE, 80, 3000))}
    random_timestamps = pipeline.forward(
        random_inputs, batch_size=BATCH_SIZE, return_timestamps=True
    )
    compile_time = time.time() - start
    logger.info(f"compiled in {compile_time}s")

    def tqdm_generate(inputs: dict, task: str, return_timestamps: bool):
        dataloader = pipeline.preprocess_batch(
            inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE
        )
        logger.info("pre-processing audio file...")
        dataloader = pool.map(identity, dataloader)
        logger.info("done post-processing")

        model_outputs = []
        start_time = time.time()
        logger.info("transcribing...")
        # iterate over our chunked audio samples - always predict timestamps to reduce hallucinations
        for batch, _ in zip(dataloader):
            model_outputs.append(
                pipeline.forward(
                    batch, batch_size=BATCH_SIZE, task=task, return_timestamps=True
                )
            )
        runtime = time.time() - start_time
        logger.info("done transcription")

        logger.info("post-processing...")
        post_processed = pipeline.postprocess(model_outputs, return_timestamps=True)
        text = post_processed["text"]
        if return_timestamps:
            timestamps = post_processed.get("chunks")
            timestamps = [
                f"[{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
                for chunk in timestamps
            ]
            text = "\n".join(str(feature) for feature in timestamps)
        logger.info("done post-processing")
        return text, runtime

    def transcribe_chunked_audio(inputs, task, return_timestamps):
        logger.info("loading audio file...")
        if inputs is None:
            logger.warning("No audio file")
            raise ValueError(
                "No audio file submitted! Please upload an audio file before submitting your request."
            )
        file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
        if file_size_mb > FILE_LIMIT_MB:
            logger.warning("Max file size exceeded")
            raise ValueError(
                f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
            )

        with open(inputs, "rb") as f:
            inputs = f.read()

        inputs = ffmpeg_read(inputs, pipeline.feature_extractor.sampling_rate)
        inputs = {
            "array": inputs,
            "sampling_rate": pipeline.feature_extractor.sampling_rate,
        }
        logger.info("done loading")
        text, runtime = tqdm_generate(
            inputs, task=task, return_timestamps=return_timestamps
        )
        return text, runtime
