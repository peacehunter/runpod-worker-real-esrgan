import runpod
from rp_handler import handler

runpod.serverless.start({"handler": handler})
