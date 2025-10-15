# To run this file:
# 1. Ensure the 'templates' folder exists with 'index.html' inside it.
# 2. Run from terminal: uvicorn sonify_server:app --reload

import asyncio
import os
import uuid
import json
import struct
import gzip
import random
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import subprocess
import logging
import numpy as np

# --- Basic Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
logging.basicConfig(level=logging.INFO)


# --- FFmpeg Filter Building Logic ---

def _build_aecho_filter(params: dict) -> str:
    """Echo with clamped ranges to stay within FFmpeg limits"""
    in_gain = min(1.0, max(0.0, float(params.get('in_gain', 0.8))))
    out_gain = min(1.0, max(0.0, float(params.get('out_gain', 0.9))))
    delay = int(params.get('delay', 1000))
    decay = min(1.0, max(0.0, float(params.get('decay', 0.3))))
    return f"aecho={in_gain}:{out_gain}:{delay}:{decay}"


def _build_bass_filter(params: dict) -> str:
    """Bass boost with extreme gain ranges for heavy distortion"""
    gain = int(params.get('gain', 5))
    return f"bass=g={gain}:f=110"

def _build_treble_filter(params: dict) -> str:
    """Treble boost with extreme gain ranges for harsh artifacts"""
    gain = int(params.get('gain', 5))
    return f"treble=g={gain}:f=5000"

def _build_phaser_filter(params: dict) -> str:
    # ... (no changes)
    in_gain = float(params.get('in_gain', 0.8))
    out_gain = float(params.get('out_gain', 0.7))
    delay = float(params.get('delay', 3.0))
    decay = float(params.get('decay', 0.4))
    return f"aphaser=in_gain={in_gain}:out_gain={out_gain}:delay={delay}:decay={decay}"

def _build_flanger_filter(params: dict) -> str:
    """Flanger effect creates sweeping comb filter artifacts"""
    delay = float(params.get('delay', 2.0))
    depth = float(params.get('depth', 10.0))
    regen = float(params.get('regen', 0.5))
    width = float(params.get('width', 70.0))
    speed = float(params.get('speed', 0.5))
    shape = params.get('shape', 'sine')
    return f"flanger=delay={delay}:depth={depth}:regen={regen}:width={width}:speed={speed}:shape={shape}"

def _build_vibrato_filter(params: dict) -> str:
    """Vibrato creates modulation artifacts that can pixelate images"""
    freq = float(params.get('freq', 5.0))
    depth = float(params.get('depth', 0.5))
    return f"vibrato=f={freq}:d={depth}"

def _build_distortion_filter(params: dict) -> str:
    """Heavy distortion for crushing pixel values"""
    level_in = float(params.get('level_in', 1.0))
    level_out = float(params.get('level_out', 1.0))
    amount = float(params.get('amount', 0.5))
    return f"aexciter=level_in={level_in}:level_out={level_out}:amount={amount}"

def _build_pitch_shift_filter(params: dict) -> str:
    """Pitch shifting scrambles data by resampling"""
    shift = int(params.get('shift', 0))
    return f"apitchshift=shift={shift}"

def _build_stretch_filter(params: dict) -> str:
    """Time stretching creates weird artifacts by changing playback speed"""
    tempo = float(params.get('tempo', 0.5))
    return f"atempo={tempo}"

def _build_compand_filter(params: dict) -> str:
    """Compressor can crush dynamic range and create harsh artifacts"""
    attack = float(params.get('attack', 0.2))
    decay = float(params.get('decay', 0.8))
    soft_knee = float(params.get('soft_knee', 0.01))
    threshold = float(params.get('threshold', -20))
    ratio = float(params.get('ratio', 20))
    makeup = float(params.get('makeup', 8))
    return f"compand=attack={attack}:decay={decay}:soft_knee={soft_knee}:threshold={threshold}:ratio={ratio}:makeup={makeup}"

def _build_highpass_filter(params: dict) -> str:
    """High pass filter removes low frequency data"""
    frequency = int(params.get('frequency', 500))
    return f"highpass=f={frequency}"

def _build_lowpass_filter(params: dict) -> str:
    """Low pass filter removes high frequency data"""
    frequency = int(params.get('frequency', 5000))
    return f"lowpass=f={frequency}"

# --- Data Corruption Effects ---

def _apply_bit_crush(data: np.ndarray, params: dict) -> np.ndarray:
    """Reduce bit depth for quantization artifacts"""
    bits = int(params.get('bits', 4))
    max_val = (2 ** bits) - 1
    # Quantize to fewer bits
    quantized = np.round(data / 255.0 * max_val) / max_val * 255.0
    return np.clip(quantized, 0, 255).astype(np.uint8)

def _apply_data_mosh(data: np.ndarray, params: dict) -> np.ndarray:
    """Shuffle byte sequences for data moshing effects"""
    intensity = float(params.get('intensity', 0.1))
    data_copy = data.copy()

    # Random byte shuffling
    num_swaps = int(len(data_copy) * intensity)
    for _ in range(num_swaps):
        idx1 = random.randint(0, len(data_copy) - 1)
        idx2 = random.randint(0, len(data_copy) - 1)
        data_copy[idx1], data_copy[idx2] = data_copy[idx2], data_copy[idx1]

    return data_copy

def _apply_glitch_blocks(data: np.ndarray, params: dict) -> np.ndarray:
    """Create block-based corruption artifacts"""
    block_size = int(params.get('block_size', 64))
    intensity = float(params.get('intensity', 0.05))
    data_copy = data.copy()

    total_bytes = len(data_copy)
    num_blocks = int(total_bytes / (block_size * 3))  # RGB blocks
    num_corrupt = int(num_blocks * intensity)

    for _ in range(num_corrupt):
        block_idx = random.randint(0, num_blocks - 1)
        start_byte = block_idx * block_size * 3

        if start_byte + block_size * 3 <= total_bytes:
            # Corrupt this block with random data
            corrupt_data = np.random.randint(0, 256, block_size * 3, dtype=np.uint8)
            data_copy[start_byte:start_byte + block_size * 3] = corrupt_data

    return data_copy

def _apply_channel_swap(data: np.ndarray, params: dict) -> np.ndarray:
    """Randomly swap RGB color channels"""
    intensity = float(params.get('intensity', 0.3))
    data_copy = data.copy()

    # Reshape to RGB channels
    pixels = data_copy.reshape(-1, 3)

    # Random channel swapping
    num_swaps = int(len(pixels) * intensity)
    for _ in range(num_swaps):
        pixel_idx = random.randint(0, len(pixels) - 1)
        # Randomly permute channels
        channels = list(pixels[pixel_idx])
        random.shuffle(channels)
        pixels[pixel_idx] = channels

    return data_copy

def _build_bit_crush_filter(params: dict) -> str:
    """Bit crushing effect - handled as custom processing"""
    return "custom_bit_crush"

def _build_data_mosh_filter(params: dict) -> str:
    """Data moshing effect - handled as custom processing"""
    return "custom_data_mosh"

def _build_glitch_blocks_filter(params: dict) -> str:
    """Glitch block corruption - handled as custom processing"""
    return "custom_glitch_blocks"

def _build_channel_swap_filter(params: dict) -> str:
    """RGB channel swapping - handled as custom processing"""
    return "custom_channel_swap"

FILTER_BUILDERS = {
    "aecho": _build_aecho_filter,
    "bass": _build_bass_filter,
    "treble": _build_treble_filter,
    "phaser": _build_phaser_filter,
    "flanger": _build_flanger_filter,
    "vibrato": _build_vibrato_filter,
    "distortion": _build_distortion_filter,
    "pitch_shift": _build_pitch_shift_filter,
    "stretch": _build_stretch_filter,
    "compand": _build_compand_filter,
    "highpass": _build_highpass_filter,
    "lowpass": _build_lowpass_filter,
    "bit_crush": _build_bit_crush_filter,
    "data_mosh": _build_data_mosh_filter,
    "glitch_blocks": _build_glitch_blocks_filter,
    "channel_swap": _build_channel_swap_filter,
}

# --- API Endpoints ---

@app.post("/process-chain/")
async def process_filter_chain(
    pixel_data: UploadFile = File(...),
    filter_chain: str = Form(...)
):
    """
    BATCH FILTER CHAIN PROCESSING: Process multiple FFmpeg filters in one pass
    This endpoint takes a chain of FFmpeg filters and applies them all at once,
    which is much faster than making separate server calls for each filter.
    Only handles FFmpeg filters - custom effects are processed in the browser.
    """
    try:
        logging.info(f"Processing filter chain: {filter_chain}")

        # Read the uploaded audio data (already converted from pixels)
        contents = await pixel_data.read()

        # Try to decompress if the client sent gzip-compressed data
        try:
            contents = gzip.decompress(contents)
            logging.info("Decompressed gzip data")
        except:
            logging.info("Processing uncompressed data")

        original_length = len(contents)
        logging.info(f"Received {original_length} bytes of audio data for batch processing.")

        # Convert bytes to numpy array (already in float32 format)
        float_array = np.frombuffer(contents, dtype=np.float32)

        # Build FFmpeg command with combined filter chain
        ffmpeg_command = [
            'ffmpeg', '-y',
            '-f', 'f32le',
            '-ar', '44100',
            '-ac', '1',
            '-i', 'pipe:0',
            '-af', filter_chain,  # Apply the entire filter chain at once
            '-f', 'f32le',
            'pipe:1'
        ]

        logging.info(f"Executing FFmpeg with batch filter chain...")
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Send data to FFmpeg and get result
        filtered_float_bytes, stderr = await process.communicate(input=float_array.tobytes())

        if process.returncode != 0:
            error_message = stderr.decode()
            logging.error(f"FFmpeg Error: {error_message}")
            raise HTTPException(status_code=500, detail=f"FFmpeg batch processing failed: {error_message}")

        logging.info("FFmpeg batch processing successful.")

        # Return the processed audio data (will be converted back to pixels in browser)
        return Response(content=filtered_float_bytes, media_type="application/octet-stream")

    except Exception as e:
        logging.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")


@app.post("/process-image/")
async def process_image_data(
    pixel_data: UploadFile = File(...),
    filter_name: str = Form(...),
    filter_params: str = Form(...)
):
    if filter_name not in FILTER_BUILDERS:
        raise HTTPException(status_code=400, detail="Invalid filter name provided.")

    try:
        params = json.loads(filter_params)
        filter_string = FILTER_BUILDERS[filter_name](params)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter parameters: {e}")

    try:
        # Handle compressed data
        contents = await pixel_data.read()

        # Try to decompress if it's gzip compressed
        try:
            contents = gzip.decompress(contents)
            logging.info("Decompressed gzip data")
        except:
            logging.info("Processing uncompressed data")

        original_length = len(contents)
        logging.info(f"Received {original_length} bytes of pixel data to process.")

        # Convert bytes to numpy array
        pixel_array = np.frombuffer(contents, dtype=np.uint8)

        # Check if this is a custom data corruption filter
        if filter_string.startswith("custom_"):
            logging.info(f"Applying custom data corruption filter: {filter_name}")

            # Apply custom corruption effects
            if filter_name == "bit_crush":
                filtered_pixel_array = _apply_bit_crush(pixel_array, params)
            elif filter_name == "data_mosh":
                filtered_pixel_array = _apply_data_mosh(pixel_array, params)
            elif filter_name == "glitch_blocks":
                filtered_pixel_array = _apply_glitch_blocks(pixel_array, params)
            elif filter_name == "channel_swap":
                filtered_pixel_array = _apply_channel_swap(pixel_array, params)
            else:
                filtered_pixel_array = pixel_array.copy()

            logging.info(f"Custom corruption filter {filter_name} applied successfully.")

        else:
            # Standard FFmpeg audio processing
            logging.info("Converting pixel data to 32-bit float audio stream using NumPy...")

            # Vectorized conversion to float audio range (-1.0 to 1.0)
            float_array = (pixel_array.astype(np.float32) / 127.5) - 1.0

            # Convert to bytes for FFmpeg
            float_bytes = float_array.astype(np.float32).tobytes()

            # Use memory-based FFmpeg processing with pipes
            ffmpeg_command = [
                'ffmpeg', '-y', '-f', 'f32le', '-ar', '44100', '-ac', '1',
                '-i', 'pipe:0',
                '-af', filter_string,
                '-f', 'f32le', 'pipe:1'
            ]

            logging.info(f"Executing FFmpeg with pipe processing...")
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Send data to FFmpeg and get result
            filtered_float_bytes, stderr = await process.communicate(input=float_bytes)

            if process.returncode != 0:
                error_message = stderr.decode()
                logging.error(f"FFmpeg Error: {error_message}")
                raise HTTPException(status_code=500, detail=f"FFmpeg processing failed: {error_message}")

            logging.info("FFmpeg processing successful.")

            logging.info("Converting filtered audio back to pixel data using NumPy...")

            # Convert filtered bytes back to numpy array
            filtered_float_array = np.frombuffer(filtered_float_bytes, dtype=np.float32)

            # Vectorized conversion back to byte range (0-255)
            filtered_pixel_array = ((filtered_float_array + 1.0) * 127.5).astype(np.uint8)

        # Ensure correct length
        if len(filtered_pixel_array) > original_length:
            filtered_pixel_array = filtered_pixel_array[:original_length]
        elif len(filtered_pixel_array) < original_length:
            # Pad with zeros if needed
            padding = original_length - len(filtered_pixel_array)
            filtered_pixel_array = np.pad(filtered_pixel_array, (0, padding), 'constant')

        return Response(content=filtered_pixel_array.tobytes(), media_type="application/octet-stream")

    except Exception as e:
        logging.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.get("/worker.js")
async def get_worker():
    """Serve the Web Worker script."""
    with open("templates/worker.js", "r") as f:
        content = f.read()
    return Response(content=content, media_type="application/javascript")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
