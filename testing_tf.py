import tensorflow as tf
import os

# Check CUDA and GPU status
cuda_version = os.popen('nvcc --version').read()
nvidia_smi = os.popen('nvidia-smi').read()

print("CUDA Version:\n", cuda_version)
print("NVIDIA-SMI Output:\n", nvidia_smi)

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# List available devices
physical_devices = tf.config.list_physical_devices('GPU')
print("Physical GPUs:", physical_devices)

# Set memory growth for GPUs
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth set.")
else:
    print("No GPUs found. Please ensure you have installed TensorFlow with GPU support, and that your GPU is properly set up.")
