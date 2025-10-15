# =============================================================================
# IMAGE SONIFICATION & GLITCH ART GENERATOR
# =============================================================================
#
# This is the most unhinged code you'll ever see. We're taking images, converting
# them to audio, applying audio filters, then converting back to images. Why?
# Because audio filters create beautiful corruption artifacts when applied to
# visual data. It's like using a chainsaw to perform surgery.
#
# RUN INSTRUCTIONS:
# 1. Make sure 'templates' folder exists with 'index.html' inside it
# 2. Install dependencies: pip install fastapi uvicorn numpy aiofiles
# 3. Also need FFmpeg installed on your system (sudo apt install ffmpeg or brew install ffmpeg)
# 4. Run: uvicorn sonify_prototype:app --reload
# 5. Open browser to http://localhost:8000
# 6. Upload an image and apply "audio filters" to create glitch art
#
# THE MAGIC:
# - Image pixels (RGB values 0-255) get converted to audio samples (-1.0 to 1.0)
# - FFmpeg audio filters treat this "audio" data in weird ways
# - Echo becomes pixel echo, bass boost becomes color saturation boost
# - The filtered audio gets converted back to pixel values
# - Result: Beautifully corrupted, glitched images with audio filter artifacts
# =============================================================================

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

# =============================================================================
# BASIC FASTAPI SETUP
# =============================================================================
# FastAPI is our web framework - it handles HTTP requests and serves the HTML
# templates and API endpoints. Nothing fancy here, just standard web server setup.

app = FastAPI()  # Create the FastAPI application instance
templates = Jinja2Templates(directory="templates")  # Template renderer for HTML
logging.basicConfig(level=logging.INFO)  # Log everything at INFO level for debugging

# =============================================================================
# AUDIO FILTER BUILDERS
# =============================================================================
# Each function builds an FFmpeg filter string. These are the same filters you'd
# use in audio production, but we're feeding them image data instead. The chaos
# that ensues is beautiful and unpredictable.

def _build_aecho_filter(params: dict) -> str:
    """
    ECHO FILTER: Creates repeating pixel patterns
    Normal echo repeats sound. We're repeating pixels. Each echo creates
    ghostly copies of image data at different "delays" (byte offsets).
    This creates weird duplication effects in the final image.
    """
    in_gain = min(1.0, max(0.0, float(params.get('in_gain', 0.8))))  # Input volume control
    out_gain = min(1.0, max(0.0, float(params.get('out_gain', 0.9))))  # Output volume control
    delay = int(params.get('delay', 1000))  # Delay in milliseconds = byte offset for echo
    decay = min(1.0, max(0.0, float(params.get('decay', 0.3))))  # How quickly echo fades
    return f"aecho={in_gain}:{out_gain}:{delay}:{decay}"  # FFmpeg aecho filter syntax


def _build_bass_filter(params: dict) -> str:
    """
    BASS BOOST: Amplifies low-frequency pixel data
    In audio, bass affects low frequencies. In pixel data, "low frequencies"
    correspond to slowly changing color values (smooth gradients). This boosts
    smooth areas and can create weird color saturation effects.
    """
    gain = int(params.get('gain', 5))  # How much to boost (higher = more extreme)
    return f"bass=g={gain}:f=110"  # Boost at 110Hz (arbitrary low frequency)


def _build_treble_filter(params: dict) -> str:
    """
    TREBLE BOOST: Amplifies high-frequency pixel data
    High frequencies in audio are rapid changes. In pixels, this means
    sharp edges, noise, and fine details. This enhances noise and creates
    harsh edge enhancement artifacts.
    """
    gain = int(params.get('gain', 5))  # Boost amount (higher = more harsh)
    return f"treble=g={gain}:f=5000"  # Boost at 5kHz (arbitrary high frequency)


def _build_phaser_filter(params: dict) -> str:
    """
    PHASER: Creates sweeping comb-filter artifacts
    Phasers create moving notches in the frequency spectrum. On pixel data,
    this creates weird periodic corruption that sweeps through the image.
    Like a glitch that travels across your photo.
    """
    in_gain = float(params.get('in_gain', 0.8))  # Input gain
    out_gain = float(params.get('out_gain', 0.7))  # Output gain
    delay = float(params.get('delay', 3.0))  # Base delay for the phaser effect
    decay = float(params.get('decay', 0.4))  # How quickly the effect fades
    return f"aphaser=in_gain={in_gain}:out_gain={out_gain}:delay={delay}:decay={decay}"


def _build_flanger_filter(params: dict) -> str:
    """
    FLANGER: Creates metallic, sweeping artifacts
    Flangers are like phasers but more metallic sounding. On pixel data,
    they create weird periodic color shifts and metallic-looking artifacts.
    The sweeping motion creates animated corruption patterns.
    """
    delay = float(params.get('delay', 2.0))  # Base delay
    depth = float(params.get('depth', 10.0))  # How deep the modulation goes
    regen = float(params.get('regen', 0.5))  # Regeneration (feedback)
    width = float(params.get('width', 70.0))  # Width of the effect
    speed = float(params.get('speed', 0.5))  # Speed of the sweeping motion
    shape = params.get('shape', 'sine')  # Shape of the modulation (sine, triangle, etc.)
    return f"flanger=delay={delay}:depth={depth}:regen={regen}:width={width}:speed={speed}:shape={shape}"


def _build_vibrato_filter(params: dict) -> str:
    """
    VIBRATO: Modulates pixel values periodically
    Vibrato wobbles the pitch in audio. In pixels, this means the RGB values
    get wobbled up and down periodically. This creates weird color oscillation
    and pixelation effects across the image.
    """
    freq = float(params.get('freq', 5.0))  # Frequency of wobbling
    depth = float(params.get('depth', 0.5))  # How intense the wobbling is
    return f"vibrato=f={freq}:d={depth}"


def _build_distortion_filter(params: dict) -> str:
    """
    DISTORTION: Crushes and clips pixel values
    Audio distortion adds harmonics and clips peaks. In pixels, this means
    extreme color values get clipped to maximum/minimum, creating harsh
    contrast and color saturation artifacts. It's like turning your image
    up to 11.
    """
    level_in = float(params.get('level_in', 1.0))  # Input level (how much to boost before clipping)
    level_out = float(params.get('level_out', 1.0))  # Output level (final volume adjustment)
    amount = float(params.get('amount', 0.5))  # Amount of distortion to apply
    return f"aexciter=level_in={level_in}:level_out={level_out}:amount={amount}"


def _build_pitch_shift_filter(params: dict) -> str:
    """
    PITCH SHIFT: Resamples and stretches pixel data
    Pitch shifting resamples audio. In pixels, this means we're stretching
    or compressing the pixel data sequence, which creates weird stretching
    and compression artifacts. It's like pulling on the fabric of your image.
    """
    shift = int(params.get('shift', 0))  # Semitones to shift (positive = up, negative = down)
    return f"apitchshift=shift={shift}"


def _build_stretch_filter(params: dict) -> str:
    """
    TIME STRETCH: Changes playback speed without changing pitch
    This is pure chaos for pixel data. It stretches the pixel sequence in time,
    creating weird duplication and stretching artifacts. Low tempo creates
    smeared, dreamlike images. High tempo creates frantic, glitched madness.
    """
    tempo = float(params.get('tempo', 0.5))  # Tempo multiplier (0.5 = half speed, 2.0 = double speed)
    return f"atempo={tempo}"


def _build_compand_filter(params: dict) -> str:
    """
    COMPRESSOR: Crushes dynamic range
    Compressors make quiet sounds louder and loud sounds quieter.
    In pixels, this means bright areas get dimmer and dark areas get brighter,
    creating weird contrast flattening and harsh color artifacts. It's like
    applying aggressive auto-contrast to your image.
    """
    attack = float(params.get('attack', 0.2))  # How quickly compressor responds (in seconds)
    decay = float(params.get('decay', 0.8))  # How quickly it releases (in seconds)
    soft_knee = float(params.get('soft_knee', 0.01))  # How gradual the compression is
    threshold = float(params.get('threshold', -20))  # At what level compression starts (in dB)
    ratio = float(params.get('ratio', 20))  # How much to compress (higher = more aggressive)
    makeup = float(params.get('makeup', 8))  # Makeup gain to compensate for volume loss
    return f"compand=attack={attack}:decay={decay}:soft_knee={soft_knee}:threshold={threshold}:ratio={ratio}:makeup={makeup}"


def _build_highpass_filter(params: dict) -> str:
    """
    HIGH PASS FILTER: Removes "low frequency" pixel data
    This removes smooth gradients and slowly changing color values,
    leaving only sharp edges and rapid changes. Creates a weird
    edge-detection effect with harsh artifacts.
    """
    frequency = int(params.get('frequency', 500))  # Cutoff frequency in Hz
    return f"highpass=f={frequency}"


def _build_lowpass_filter(params: dict) -> str:
    """
    LOW PASS FILTER: Removes "high frequency" pixel data
    This removes sharp edges and noise, leaving only smooth gradients.
    Creates a weird blur/smoothing effect that can also introduce
    bizarre color bleeding artifacts.
    """
    frequency = int(params.get('frequency', 5000))  # Cutoff frequency in Hz
    return f"lowpass=f={frequency}"

# =============================================================================
# CUSTOM DATA CORRUPTION EFFECTS
# =============================================================================
# These are our own custom "filters" that directly corrupt the pixel data.
# No audio processing here - just pure, unadulterated data manipulation.

def _apply_bit_crush(data: np.ndarray, params: dict) -> np.ndarray:
    """
    BIT CRUSH: Reduces color precision
    Normal images use 8 bits per color channel (256 levels). Bit crushing
    reduces this to fewer levels (like 16 levels or 4 levels). This creates
    the classic posterization effect where smooth gradients become bands
    of solid colors. It's like drawing with a limited crayon set.
    """
    bits = int(params.get('bits', 4))  # How many bits to keep (4 = 16 colors, 1 = 2 colors)
    max_val = (2 ** bits) - 1  # Maximum value with reduced bits

    # Convert to 0-1 range, quantize to fewer levels, then convert back to 0-255
    quantized = np.round(data / 255.0 * max_val) / max_val * 255.0
    return np.clip(quantized, 0, 255).astype(np.uint8)  # Ensure values stay in valid range


def _apply_data_mosh(data: np.ndarray, params: dict) -> str:
    """
    DATA MOSH: Randomly shuffles bytes
    This is pure chaos. We randomly swap bytes throughout the image data.
    The result is unpredictable corruption that can create weird patterns,
    colors, and structural damage to the image. Like a digital poltergeist.
    """
    intensity = float(params.get('intensity', 0.1))  # How much corruption (0.1 = 10% of bytes get shuffled)
    data_copy = data.copy()  # Don't corrupt the original

    # Calculate how many bytes to shuffle
    num_swaps = int(len(data_copy) * intensity)

    for _ in range(num_swaps):
        # Pick two random positions and swap them
        idx1 = random.randint(0, len(data_copy) - 1)
        idx2 = random.randint(0, len(data_copy) - 1)
        data_copy[idx1], data_copy[idx2] = data_copy[idx2], data_copy[idx1]

    return data_copy


def _apply_glitch_blocks(data: np.ndarray, params: dict) -> np.ndarray:
    """
    GLITCH BLOCKS: Corrupts rectangular regions
    This simulates the classic digital glitch effect where blocks of data
    get corrupted or replaced with garbage. Creates rectangular regions
    of random colors and patterns scattered across the image.
    """
    block_size = int(params.get('block_size', 64))  # Size of corrupted blocks in bytes
    intensity = float(params.get('intensity', 0.05))  # How many blocks to corrupt (0.05 = 5%)
    data_copy = data.copy()

    total_bytes = len(data_copy)
    num_blocks = int(total_bytes / (block_size * 3))  # Calculate total RGB blocks
    num_corrupt = int(num_blocks * intensity)  # How many blocks to corrupt

    for _ in range(num_corrupt):
        # Pick a random block to corrupt
        block_idx = random.randint(0, num_blocks - 1)
        start_byte = block_idx * block_size * 3  # RGB = 3 bytes per pixel

        if start_byte + block_size * 3 <= total_bytes:
            # Replace this block with completely random data
            corrupt_data = np.random.randint(0, 256, block_size * 3, dtype=np.uint8)
            data_copy[start_byte:start_byte + block_size * 3] = corrupt_data

    return data_copy


def _apply_channel_swap(data: np.ndarray, params: dict) -> np.ndarray:
    """
    CHANNEL SWAP: Randomly swaps RGB color channels
    This randomly swaps the Red, Green, and Blue values in pixels.
    Creates weird color inversion and swapping effects. Sometimes you get
    blues where reds should be, greens where blues should be, etc.
    The human brain tries to make sense of the impossible colors.
    """
    intensity = float(params.get('intensity', 0.3))  # How many pixels to affect (0.3 = 30%)
    data_copy = data.copy()

    # Reshape the flat byte array into RGB pixel arrays
    pixels = data_copy.reshape(-1, 3)

    # Randomly shuffle RGB channels in random pixels
    num_swaps = int(len(pixels) * intensity)
    for _ in range(num_swaps):
        pixel_idx = random.randint(0, len(pixels) - 1)
        # Shuffle the R, G, B values in this pixel
        channels = list(pixels[pixel_idx])
        random.shuffle(channels)
        pixels[pixel_idx] = channels

    return data_copy


# =============================================================================
# CUSTOM FILTER BUILDERS (These just return special markers)
# =============================================================================
# These return "custom_*" strings to tell the main processing function
# to use our custom data corruption effects instead of FFmpeg.

def _build_bit_crush_filter(params: dict) -> str:
    """Returns marker to use custom bit crush processing"""
    return "custom_bit_crush"

def _build_data_mosh_filter(params: dict) -> str:
    """Returns marker to use custom data moshing processing"""
    return "custom_data_mosh"

def _build_glitch_blocks_filter(params: dict) -> str:
    """Returns marker to use custom glitch block processing"""
    return "custom_glitch_blocks"

def _build_channel_swap_filter(params: dict) -> str:
    """Returns marker to use custom channel swap processing"""
    return "custom_channel_swap"

# =============================================================================
# FILTER REGISTRY
# =============================================================================
# This dictionary maps filter names to their builder functions.
# When a filter name comes in from the frontend, we look it up here
# to get the corresponding function that builds the FFmpeg filter string
# or returns a custom processing marker.

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
    # Custom data corruption effects
    "bit_crush": _build_bit_crush_filter,
    "data_mosh": _build_data_mosh_filter,
    "glitch_blocks": _build_glitch_blocks_filter,
    "channel_swap": _build_channel_swap_filter,
}

# =============================================================================
# API ENDPOINTS
# =============================================================================
# These are the HTTP endpoints that the frontend calls.

@app.post("/process-image/")
async def process_image_data(
    pixel_data: UploadFile = File(...),  # The raw pixel data uploaded as a file
    filter_name: str = Form(...),        # Which filter to apply
    filter_params: str = Form(...)       # JSON string of filter parameters
):
    """
    MAIN PROCESSING ENDPOINT: Where the magic happens
    This endpoint takes uploaded image pixel data, converts it to audio,
    applies audio filters via FFmpeg, then converts back to image data.
    The result is beautifully corrupted glitch art.
    """

    # Validate that the requested filter exists
    if filter_name not in FILTER_BUILDERS:
        raise HTTPException(status_code=400, detail="Invalid filter name provided.")

    # Parse filter parameters from JSON string
    try:
        params = json.loads(filter_params)
        filter_string = FILTER_BUILDERS[filter_name](params)  # Build the FFmpeg filter string
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter parameters: {e}")

    try:
        # Read the uploaded pixel data
        contents = await pixel_data.read()

        # Try to decompress if the client sent gzip-compressed data (for faster uploads)
        try:
            contents = gzip.decompress(contents)
            logging.info("Decompressed gzip data")
        except:
            logging.info("Processing uncompressed data")

        original_length = len(contents)
        logging.info(f"Received {original_length} bytes of pixel data to process.")

        # Convert the raw bytes to a numpy array for efficient processing
        pixel_array = np.frombuffer(contents, dtype=np.uint8)

        # =======================================================================
        # CUSTOM DATA CORRUPTION PATH
        # =======================================================================
        # If the filter string starts with "custom_", we bypass FFmpeg entirely
        # and apply our custom data corruption effects directly to the pixel data.

        if filter_string.startswith("custom_"):
            logging.info(f"Applying custom data corruption filter: {filter_name}")

            # Apply the appropriate custom effect
            if filter_name == "bit_crush":
                filtered_pixel_array = _apply_bit_crush(pixel_array, params)
            elif filter_name == "data_mosh":
                filtered_pixel_array = _apply_data_mosh(pixel_array, params)
            elif filter_name == "glitch_blocks":
                filtered_pixel_array = _apply_glitch_blocks(pixel_array, params)
            elif filter_name == "channel_swap":
                filtered_pixel_array = _apply_channel_swap(pixel_array, params)
            else:
                # If somehow we don't recognize the custom filter, just return the original
                filtered_pixel_array = pixel_array.copy()

            logging.info(f"Custom corruption filter {filter_name} applied successfully.")

        # =======================================================================
        # FFMPEG AUDIO PROCESSING PATH
        # =======================================================================
        # This is where the real audio processing happens. We convert pixel data
        # to audio, send it through FFmpeg, then convert back to pixels.

        else:
            logging.info("Converting pixel data to 32-bit float audio stream using NumPy...")

            # Convert pixel values (0-255) to audio sample values (-1.0 to 1.0)
            # Formula: normalized = (value / 127.5) - 1.0
            # This maps: 0 -> -1.0, 128 -> 0.0, 255 -> 1.0
            float_array = (pixel_array.astype(np.float32) / 127.5) - 1.0

            # Convert the float array to bytes that FFmpeg can read
            float_bytes = float_array.astype(np.float32).tobytes()

            # Build the FFmpeg command
            ffmpeg_command = [
                'ffmpeg', '-y',  # -y = overwrite output files without asking
                '-f', 'f32le',   # Input format: 32-bit float little-endian
                '-ar', '44100',  # Audio sample rate: 44.1kHz (standard CD quality)
                '-ac', '1',      # Audio channels: 1 (mono)
                '-i', 'pipe:0',  # Input from stdin (pipe)
                '-af', filter_string,  # Audio filter to apply
                '-f', 'f32le',   # Output format: 32-bit float little-endian
                'pipe:1'         # Output to stdout (pipe)
            ]

            logging.info(f"Executing FFmpeg with pipe processing...")

            # Run FFmpeg as a subprocess with pipes for stdin/stdout
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdin=asyncio.subprocess.PIPE,  # We'll send audio data here
                stdout=asyncio.subprocess.PIPE,  # We'll read processed audio from here
                stderr=asyncio.subprocess.PIPE   # We'll read any error messages from here
            )

            # Send the "audio" data to FFmpeg and get the filtered result
            filtered_float_bytes, stderr = await process.communicate(input=float_bytes)

            # Check if FFmpeg failed
            if process.returncode != 0:
                error_message = stderr.decode()
                logging.error(f"FFmpeg Error: {error_message}")
                raise HTTPException(status_code=500, detail=f"FFmpeg processing failed: {error_message}")

            logging.info("FFmpeg processing successful.")
            logging.info("Converting filtered audio back to pixel data using NumPy...")

            # Convert the filtered audio bytes back to a numpy array
            filtered_float_array = np.frombuffer(filtered_float_bytes, dtype=np.float32)

            # Convert audio sample values (-1.0 to 1.0) back to pixel values (0-255)
            # Reverse formula: pixel = ((sample + 1.0) * 127.5)
            filtered_pixel_array = ((filtered_float_array + 1.0) * 127.5).astype(np.uint8)

        # =======================================================================
        # POST-PROCESSING: Ensure Correct Length
        # =======================================================================
        # FFmpeg might change the length of the data slightly due to how filters work.
        # We need to make sure the output has exactly the same number of bytes
        # as the input, otherwise the frontend can't reconstruct the image properly.

        if len(filtered_pixel_array) > original_length:
            # If we got too much data, truncate it
            filtered_pixel_array = filtered_pixel_array[:original_length]
        elif len(filtered_pixel_array) < original_length:
            # If we got too little data, pad with zeros
            padding = original_length - len(filtered_pixel_array)
            filtered_pixel_array = np.pad(filtered_pixel_array, (0, padding), 'constant')

        # =======================================================================
        # RETURN THE PROCESSED DATA
        # =======================================================================
        # Send the corrupted pixel data back to the frontend as raw bytes.
        # The frontend will reconstruct the image from this data.

        return Response(content=filtered_pixel_array.tobytes(), media_type="application/octet-stream")

    except Exception as e:
        logging.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.get("/worker.js")
async def get_worker():
    """
    SERVE WEB WORKER: The frontend uses a Web Worker for heavy processing
    (RGB extraction and image reconstruction) to avoid blocking the UI thread.
    This endpoint serves that JavaScript file.
    """
    with open("templates/worker.js", "r") as f:
        content = f.read()
    return Response(content=content, media_type="application/javascript")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    MAIN PAGE: Serve the HTML interface
    This serves the main web page where users upload images and apply filters.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# =============================================================================
# CONCLUSION
# =============================================================================
# And that's it! You now have a fully functional image sonification and glitch art
# generator. Upload any image, apply "audio filters", and watch the beautiful
# corruption unfold. Each filter creates unique artifacts based on how audio
# processing algorithms interpret visual data.
#
# The results are unpredictable, often beautiful, and always interesting.
# Welcome to the dark side of creative coding.