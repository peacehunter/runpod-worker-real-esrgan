import os
import io
import uuid
import base64
import cv2
import glob
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from PIL import Image
from schemas.input import INPUT_SCHEMA
import torch  # Import torch for CUDA memory management

# Combine both settings in a single environment variable assignment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # or try a different value


GPU_ID = 0
VOLUME_PATH = '/workspace'
TMP_PATH = f'{VOLUME_PATH}/tmp'
MODELS_PATH = f'{VOLUME_PATH}/models/ESRGAN'
GFPGAN_MODEL_PATH = f'{VOLUME_PATH}/models/GFPGAN/GFPGANv1.3.pth'
logger = RunPodLogger()

# Set a memory fraction limit and avoid fragmentation
torch.cuda.set_per_process_memory_fraction(0.8, GPU_ID)  # Limit to 80% of GPU memory
torch.backends.cudnn.benchmark = True  # Enable optimization for dynamic input sizes

# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def upscale(
        source_image_path,
        image_extension,
        model_name='RealESRGAN_x4plus',
        outscale=4,
        face_enhance=False,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        denoise_strength=0.5
):
    """
    model_name options:
        - RealESRGAN_x4plus
        - RealESRNet_x4plus
        - RealESRGAN_x4plus_anime_6B
        - RealESRGAN_x2plus
        - realesr-animevideov3
        - realesr-general-x4v3
    """
    
    model_name = model_name.split('.')[0]

    if image_extension == '.jpg':
        image_format = 'JPEG'
    elif image_extension == '.png':
        image_format = 'PNG'
    else:
        raise ValueError(f'Unsupported image type, must be either JPEG or PNG')

    # Select model and net scale based on the model name
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    model_path = os.path.join(MODELS_PATH, model_name + '.pth')
    if not os.path.isfile(model_path):
        raise Exception(f'Could not find model: {model_path}')

    dni_weight = None

    # Initialize RealESRGANer for upscaling
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=GPU_ID
    )

    # If face enhancement is enabled, initialize GFPGAN
    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    img = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise RuntimeError(f'Source image ({source_image_path}) is corrupt')

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        raise RuntimeError(e)
    else:
        result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format=image_format)
        image_data = output_buffer.getvalue()

        # Explicitly clear CUDA memory
        del upsampler
        if face_enhance:
            del face_enhancer
        torch.cuda.empty_cache()

        return base64.b64encode(image_data).decode('utf-8')

# ---------------------------------------------------------------------------- #
# Helper Functions                                                            #
# ---------------------------------------------------------------------------- #
def determine_file_extension(image_data):
    image_extension = None
    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        else:
            image_extension = '.png'
    except Exception as e:
        image_extension = '.png'
    return image_extension

# ---------------------------------------------------------------------------- #
# RunPod API Handler                                                           #
# ---------------------------------------------------------------------------- #
def upscaling_api(input):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)

    unique_id = uuid.uuid4()
    source_image_data = input['source_image']
    model_name = input['model']
    outscale = input['scale']
    face_enhance = input['face_enhance']
    tile = input['tile']
    tile_pad = input['tile_pad']
    pre_pad = input['pre_pad']
    half = input['half']

    source_image = base64.b64decode(source_image_data)
    source_file_extension = determine_file_extension(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    with open(source_image_path, 'wb') as source_file:
        source_file.write(source_image)

    try:
        result_image = upscale(
            source_image_path,
            source_file_extension,
            model_name,
            outscale,
            face_enhance,
            tile,
            tile_pad,
            pre_pad,
            half
        )
    except Exception as e:
        logger.error(f'An exception was raised: {e}')
        return {'error': traceback.format_exc(), 'refresh_worker': True}

    os.remove(source_image_path)

    return {'image': result_image}

# ---------------------------------------------------------------------------- #
# RunPod Handler Function                                                      #
# ---------------------------------------------------------------------------- #
def handler(event):
    validated_input = validate(event['input'], INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {'errors': validated_input['errors']}

    return upscaling_api(validated_input['validated_input'])

# ---------------------------------------------------------------------------- #
# Main Entry Point                                                            #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start({'handler': handler})
