import logging
import time
import uuid
from multiprocessing import Pool

import jax
import jax.numpy as jnp
import numpy as np
import requests
from jax.experimental.compilation_cache import compilation_cache as cc
from modal import (
    Dict,
    Image,
    SharedVolume,
    Stub,
    gpu,
    method,
    web_endpoint,
)
from transformers.pipelines.audio_utils import ffmpeg_read
from whisper_jax import FlaxWhisperPipline


volume = SharedVolume().persist("whisper-cache-volume")

api_image = (
    Image.from_dockerhub(
        # CUDA and CUDNN required to install JAX
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        setup_dockerfile_commands=[
            "RUN apt-get update -y && apt-get install -y git python3-pip",
            "RUN ln -s /usr/bin/python3 /usr/bin/python",
        ],
    )
    .pip_install(
        "ffmpeg-python",
        "jax[cuda12_local]",
        "https://github.com/sanchit-gandhi/whisper-jax/archive/main.tar.gz",
        "requests",
    )
    .apt_install("ffmpeg")
    .run_commands(
        "echo hi i am modal",
        "nvcc --version",
        "pip install --upgrade jax jaxlib[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        gpu=True,
    )
)

stub = Stub(
    "speedy-whisper-api",
    image=api_image,
)

stub.running_jobs = Dict()
cc.initialize_cache("./jax_cache")  # TODO: Move to Modal SharedVolume
checkpoint = "openai/whisper-large-v2"

BATCH_SIZE = 32
CHUNK_LENGTH_S = 30
NUM_PROC = 32
FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 7200  # limit to 2 hour YouTube files

logger = logging.getLogger("speedy-whisper-api")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


def identity(batch):
    return batch


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
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


@stub.function(cpu=4, gpu=gpu.A100(memory=20), timeout=900)
@web_endpoint()
def run_me():
    print("I also update on file edit!")
    whisper = WhisperSingleton.getInstance()
    result, runtime = whisper.transcribe(
        "https://filebin.net/uxupfi7d18384rdw/1_Navy_Federal_Credit_Union_US-BillSimmons__Midroll___1_.mp3",
        "transcribe",
        "false",
    )
    print(result, runtime)


class WhisperSingleton:
    _instance = None

    @staticmethod
    def getInstance():
        if WhisperSingleton._instance is None:
            WhisperSingleton._instance = Whisper()
            WhisperSingleton._instance.__enter__()
        return WhisperSingleton._instance

    @staticmethod
    def cleanup():
        if WhisperSingleton._instance is not None:
            WhisperSingleton._instance.__exit__(None, None, None)
            WhisperSingleton._instance = None


@stub.cls(cpu=4, gpu=gpu.A100(memory=20), timeout=900, shared_volumes={"/jax_cache": volume})
class Whisper:
    # Code below executes on container start
    # https://modal.com/docs/guide/lifecycle-functions#__enter__-and-__aenter__
    def __enter__(self):
        print(jax.devices())
        self.pipeline = FlaxWhisperPipline(checkpoint, dtype=jnp.bfloat16, batch_size=BATCH_SIZE)
        self.pool = Pool(NUM_PROC)

        # Pre-compile Whisper-JAX code to container architecture. Subsequent function
        # calls will call this cached/compiled code which will be super-fast.
        logger.info("Compiling forward call...")
        print("Compiling forward call...")
        start = time.time()
        random_inputs = {"input_features": np.ones((BATCH_SIZE, 80, 3000))}
        self.pipeline.forward(random_inputs, batch_size=BATCH_SIZE, return_timestamps=True)
        compile_time = time.time() - start
        logger.info(f"Compiled in {compile_time}s")
        print(f"Compiled in {compile_time}s")

    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.terminate()
        self.pool.join()

    def generate_transcription(self, inputs: dict, task: str, return_timestamps: bool):
        dataloader = self.pipeline.preprocess_batch(inputs, chunk_length_s=CHUNK_LENGTH_S, batch_size=BATCH_SIZE)
        logger.info("pre-processing audio file...")
        dataloader = self.pool.map(identity, dataloader)
        logger.info("done post-processing")

        model_outputs = []
        start_time = time.time()
        logger.info("transcribing...")
        # iterate over our chunked audio samples - always predict timestamps to reduce hallucinations
        for batch, _ in zip(dataloader):
            model_outputs.append(
                self.pipeline.forward(batch, batch_size=BATCH_SIZE, task=task, return_timestamps=True)
            )
        runtime = time.time() - start_time
        logger.info("done transcription")

        logger.info("post-processing...")
        post_processed = self.pipeline.postprocess(model_outputs, return_timestamps=True)
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

    @method()
    def transcribe(self, audio_url, task, return_timestamps):
        job_id = str(uuid.uuid4())
        logger.info(f"Starting job {job_id}")

        logger.info(f"{job_id}: Downloading file...")
        response = requests.get(audio_url)
        if response.status_code != 200:
            logger.warning("Failed to download audio file")
            raise ValueError("Failed to download audio file from URL. Please check the URL and try again.")
        logger.info(f"{job_id}: Download complete.")
        file = response.content

        # Chunk file using ffmpeg
        logger.info(f"{job_id}: Chunking file...")

        try:
            inputs = ffmpeg_read(file, self.pipeline.feature_extractor.sampling_rate)
        except Exception:
            logger.warning(f"{job_id}: Failed to chunk audio file.")
            raise ValueError(
                "Failed to chunk audio file using ffmpeg. Please ensure the file is a valid audio file and try again."
            )
        inputs = {
            "array": inputs,
            "sampling_rate": self.pipeline.feature_extractor.sampling_rate,
        }
        logger.info(f"{job_id}: Chunked audio file.")
        logger.info(f"{job_id}: Transcribing using Whisper-JAX...")
        text, runtime = Whisper().generate_transcription(inputs, task=task, return_timestamps=return_timestamps)
        logger.info(f"{job_id}: Transcription complete.")
        return text, runtime
