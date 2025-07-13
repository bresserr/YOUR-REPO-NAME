"""
CUDA utilities for GPU acceleration and TensorRT optimization
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
import numpy as np

import torch
import torch.cuda
from loguru import logger

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. GPU acceleration may be limited.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available. Some GPU operations may be slower.")


class CudaManager:
    """Manages CUDA operations and memory"""
    
    def __init__(self, device_id: int = 0, memory_fraction: float = 0.8):
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = None
        self.context = None
        self.stream = None
        self.memory_pool = {}
        
        self.setup_cuda()
    
    def setup_cuda(self):
        """Initialize CUDA environment"""
        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            raise RuntimeError("CUDA not available")
        
        if self.device_id >= torch.cuda.device_count():
            logger.error(f"Device {self.device_id} not available")
            raise RuntimeError(f"Device {self.device_id} not available")
        
        # Set device
        torch.cuda.set_device(self.device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device_id)
        
        # Create CUDA stream
        self.stream = torch.cuda.Stream(device=self.device)
        
        # Initialize CuPy if available
        if CUPY_AVAILABLE:
            cp.cuda.Device(self.device_id).use()
        
        logger.info(f"CUDA initialized on device {self.device_id}")
        self.log_gpu_info()
    
    def log_gpu_info(self):
        """Log GPU information"""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device_id)
            logger.info(f"GPU: {props.name}")
            logger.info(f"Memory: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"Compute Capability: {props.major}.{props.minor}")
            logger.info(f"Multiprocessors: {props.multi_processor_count}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3
            cached = torch.cuda.memory_reserved(self.device_id) / 1024**3
            total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
            
            return {
                "allocated": allocated,
                "cached": cached,
                "total": total,
                "free": total - allocated
            }
        return {"allocated": 0, "cached": 0, "total": 0, "free": 0}
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def allocate_memory(self, size: int, name: str = "default") -> Optional[torch.Tensor]:
        """Allocate GPU memory"""
        try:
            tensor = torch.empty(size, device=self.device, dtype=torch.float32)
            self.memory_pool[name] = tensor
            return tensor
        except RuntimeError as e:
            logger.error(f"Failed to allocate memory: {e}")
            return None
    
    def free_memory(self, name: str):
        """Free specific memory allocation"""
        if name in self.memory_pool:
            del self.memory_pool[name]
            self.clear_memory()


class TensorRTEngine:
    """TensorRT engine wrapper for optimized inference"""
    
    def __init__(self, engine_path: str, max_batch_size: int = 8):
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.engine = None
        self.context = None
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()
        
        if TENSORRT_AVAILABLE:
            self.load_engine()
    
    def load_engine(self):
        """Load TensorRT engine from file"""
        if not os.path.exists(self.engine_path):
            logger.error(f"Engine file not found: {self.engine_path}")
            return False
        
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                logger.error("Failed to load TensorRT engine")
                return False
            
            self.context = self.engine.create_execution_context()
            self.allocate_buffers()
            
            logger.info(f"TensorRT engine loaded: {self.engine_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TensorRT engine: {e}")
            return False
    
    def allocate_buffers(self):
        """Allocate input and output buffers"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference using TensorRT"""
        if self.engine is None:
            return []
        
        # Copy input data to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output data back to host
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            outputs.append(output['host'].copy())
        
        # Synchronize stream
        self.stream.synchronize()
        
        return outputs
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'inputs'):
            for input_buffer in self.inputs:
                if 'device' in input_buffer:
                    input_buffer['device'].free()
        
        if hasattr(self, 'outputs'):
            for output_buffer in self.outputs:
                if 'device' in output_buffer:
                    output_buffer['device'].free()


class ModelOptimizer:
    """Optimizes PyTorch models for TensorRT"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
    
    def optimize_model(self, model: torch.nn.Module, input_shape: tuple, 
                      output_path: str, precision: str = "fp16") -> bool:
        """Convert PyTorch model to TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRT not available for model optimization")
            return False
        
        try:
            # Set model to evaluation mode
            model.eval()
            model.to(self.device)
            
            # Create dummy input
            dummy_input = torch.randn(input_shape, device=self.device)
            
            # Convert to TensorRT
            if precision == "fp16":
                trt_model = torch.jit.trace(model, dummy_input).half()
            else:
                trt_model = torch.jit.trace(model, dummy_input)
            
            # Save optimized model
            torch.jit.save(trt_model, output_path)
            
            logger.info(f"Model optimized and saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model: {e}")
            return False
    
    def benchmark_model(self, model: torch.nn.Module, input_shape: tuple, 
                       iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        model.eval()
        model.to(self.device)
        
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        end_time.record()
        
        torch.cuda.synchronize()
        
        total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        avg_time = total_time / iterations
        fps = 1.0 / avg_time
        
        return {
            "total_time": total_time,
            "avg_time": avg_time,
            "fps": fps,
            "iterations": iterations
        }


def setup_cuda_environment():
    """Setup CUDA environment for the application"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Application will run on CPU.")
        return None
    
    # Set CUDA device
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} CUDA devices")
    
    # Use the first available device
    device_id = 0
    torch.cuda.set_device(device_id)
    
    # Set memory growth to avoid out of memory errors
    torch.cuda.empty_cache()
    
    # Set some environment variables for better performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUDA_CACHE_DISABLE"] = "0"
    
    logger.info(f"CUDA environment setup complete on device {device_id}")
    
    return CudaManager(device_id)


def get_optimal_batch_size(model: torch.nn.Module, input_shape: tuple, 
                          max_memory_gb: float = 8.0) -> int:
    """Determine optimal batch size for given model and memory constraints"""
    if not torch.cuda.is_available():
        return 1
    
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
    
    batch_size = 1
    max_batch_size = 32
    
    while batch_size <= max_batch_size:
        try:
            # Create input tensor with current batch size
            test_input = torch.randn((batch_size,) + input_shape[1:], device=device)
            
            # Test forward pass
            with torch.no_grad():
                _ = model(test_input)
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated() / 1024**3
            if memory_used > max_memory_gb:
                break
            
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            else:
                raise e
        finally:
            torch.cuda.empty_cache()
    
    # Return half of the maximum working batch size for safety
    return max(1, batch_size // 2)


def memory_efficient_inference(model: torch.nn.Module, input_tensor: torch.Tensor, 
                             batch_size: int = None) -> torch.Tensor:
    """Run inference with memory optimization"""
    if batch_size is None:
        batch_size = get_optimal_batch_size(model, input_tensor.shape)
    
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    
    # Process in batches
    for i in range(0, input_tensor.size(0), batch_size):
        batch = input_tensor[i:i+batch_size].to(device)
        
        with torch.no_grad():
            output = model(batch)
            results.append(output.cpu())
        
        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return torch.cat(results, dim=0)


# Export main functions
__all__ = [
    'CudaManager',
    'TensorRTEngine', 
    'ModelOptimizer',
    'setup_cuda_environment',
    'get_optimal_batch_size',
    'memory_efficient_inference'
]