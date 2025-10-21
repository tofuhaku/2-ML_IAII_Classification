"""
GPU Configuration and Optimization Script
Checks GPU availability and optimizes settings for The Simpsons Character Recognition
"""

import torch
import os
import sys

def check_gpu_availability():
    """Check GPU availability and configuration"""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)

    # Basic CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("\nCUDA is not available!")
        print("\nPossible solutions:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA toolkit")
        print("3. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False

    # Detailed GPU information
    print(f"\n✅ CUDA is available!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    # GPU details
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processors: {props.multi_processor_count}")

    return True

def optimize_gpu_settings():
    """Set optimal GPU settings for training"""
    if not torch.cuda.is_available():
        return

    print("\n" + "=" * 60)
    print("GPU Optimization Settings")
    print("=" * 60)

    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("✅ Enabled cuDNN benchmark mode")

    # Set memory growth (helps prevent OOM)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
        print("✅ Cleared GPU cache")

    # Set optimal number of threads
    optimal_threads = min(8, os.cpu_count())
    torch.set_num_threads(optimal_threads)
    print(f"✅ Set number of threads: {optimal_threads}")

def benchmark_gpu():
    """Run a simple GPU benchmark"""
    if not torch.cuda.is_available():
        print("\n❌ Cannot benchmark - GPU not available")
        return

    print("\n" + "=" * 60)
    print("GPU Benchmark Test")
    print("=" * 60)

    device = torch.device('cuda')

    # Create test tensors
    size = 4096
    print(f"Creating {size}x{size} tensors...")

    import time

    # CPU benchmark
    start_time = time.time()
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU computation time: {cpu_time:.3f} seconds")

    # GPU benchmark
    start_time = time.time()
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    torch.cuda.synchronize()  # Wait for GPU operations to complete

    compute_start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()  # Wait for computation to complete
    gpu_time = time.time() - compute_start

    total_gpu_time = time.time() - start_time
    print(f"GPU computation time: {gpu_time:.3f} seconds")
    print(f"GPU total time (including data transfer): {total_gpu_time:.3f} seconds")

    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.1f}x faster than CPU")

    # Memory usage
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory cached: {cached:.2f} GB")

def recommend_batch_size():
    """Recommend optimal batch size based on GPU memory"""
    if not torch.cuda.is_available():
        print("\n📝 Recommended batch size for CPU: 16-32")
        return

    print("\n" + "=" * 60)
    print("Batch Size Recommendations")
    print("=" * 60)

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Rough estimates based on ResNet50 + data
    if total_memory >= 24:
        recommended_batch = 128
        print(f"🔥 High-end GPU ({total_memory:.0f}GB): Batch size {recommended_batch}")
    elif total_memory >= 12:
        recommended_batch = 64
        print(f"🚀 Mid-range GPU ({total_memory:.0f}GB): Batch size {recommended_batch}")
    elif total_memory >= 8:
        recommended_batch = 32
        print(f"⚡ Entry-level GPU ({total_memory:.0f}GB): Batch size {recommended_batch}")
    else:
        recommended_batch = 16
        print(f"💡 Low memory GPU ({total_memory:.0f}GB): Batch size {recommended_batch}")

    print(f"\nTo use this batch size, modify config.py:")
    print(f"BATCH_SIZE = {recommended_batch}")

    return recommended_batch

def test_model_loading():
    """Test loading a simple model on GPU"""
    if not torch.cuda.is_available():
        print("\n❌ Cannot test model loading - GPU not available")
        return

    print("\n" + "=" * 60)
    print("Model Loading Test")
    print("=" * 60)

    try:
        from torchvision import models
        device = torch.device('cuda')

        print("Loading ResNet50 on GPU...")
        model = models.resnet50(pretrained=False)
        model = model.to(device)

        print("✅ Model loaded successfully on GPU")

        # Test forward pass
        dummy_input = torch.randn(4, 3, 224, 224, device=device)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✅ Forward pass successful - Output shape: {output.shape}")

        # Memory usage after model loading
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"Memory used by model: {allocated:.2f} GB")

    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main function"""
    print("🚀 GPU Configuration and Optimization for The Simpsons Character Recognition")

    # Check GPU availability
    gpu_available = check_gpu_availability()

    if gpu_available:
        # Optimize settings
        optimize_gpu_settings()

        # Run benchmark
        benchmark_gpu()

        # Recommend batch size
        recommend_batch_size()

        # Test model loading
        test_model_loading()

        print("\n" + "=" * 60)
        print("✅ GPU setup completed successfully!")
        print("You can now run training with GPU acceleration:")
        print("python main.py --mode train")
        print("=" * 60)

    else:
        print("\n" + "=" * 60)
        print("❌ GPU setup failed!")
        print("The system will fall back to CPU training (much slower)")
        print("=" * 60)

if __name__ == "__main__":
    main()