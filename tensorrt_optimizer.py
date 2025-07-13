import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import time

# TensorRT imports (with fallback for systems without TensorRT)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TensorRT not available. Falling back to PyTorch inference.")

logger = logging.getLogger(__name__)

class TensorRTOptimizer:
    """Optimizes models using TensorRT for faster inference"""
    
    def __init__(self, use_fp16: bool = True, max_batch_size: int = 1):
        self.use_fp16 = use_fp16
        self.max_batch_size = max_batch_size
        self.trt_available = TRT_AVAILABLE
        self.engines = {}
        self.contexts = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not self.trt_available:
            logger.warning("TensorRT not available. Using PyTorch inference.")
        else:
            logger.info(f"TensorRT optimizer initialized with FP16: {use_fp16}")
    
    def optimize_model(self, model: torch.nn.Module, input_shapes: Dict[str, Tuple[int, ...]], 
                      model_name: str, save_path: Optional[str] = None) -> bool:
        """Optimize a PyTorch model using TensorRT"""
        try:
            if not self.trt_available:
                logger.warning("TensorRT not available. Skipping optimization.")
                return False
            
            # Export model to ONNX first
            onnx_path = f"{model_name}.onnx"
            success = self._export_to_onnx(model, input_shapes, onnx_path)
            
            if not success:
                return False
            
            # Convert ONNX to TensorRT
            engine_path = f"{model_name}.trt"
            success = self._convert_onnx_to_trt(onnx_path, engine_path, input_shapes)
            
            if success and save_path:
                # Move engine to desired location
                os.rename(engine_path, save_path)
                engine_path = save_path
            
            # Load the engine
            if success:
                self._load_engine(engine_path, model_name)
            
            # Cleanup temporary files
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Error optimizing model {model_name}: {e}")
            return False
    
    def _export_to_onnx(self, model: torch.nn.Module, input_shapes: Dict[str, Tuple[int, ...]], 
                       onnx_path: str) -> bool:
        """Export PyTorch model to ONNX format"""
        try:
            model.eval()
            
            # Create dummy inputs
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                dummy_inputs[name] = torch.randn(shape).to(self.device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                dynamic_axes={name: {0: 'batch_size'} for name in dummy_inputs.keys()}
            )
            
            logger.info(f"Model exported to ONNX: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            return False
    
    def _convert_onnx_to_trt(self, onnx_path: str, engine_path: str, 
                           input_shapes: Dict[str, Tuple[int, ...]]) -> bool:
        """Convert ONNX model to TensorRT engine"""
        try:
            if not self.trt_available:
                return False
            
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(trt_logger)
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Create ONNX parser
            parser = trt.OnnxParser(network, trt_logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    return False
            
            # Create builder config
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            if self.use_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Set optimization profiles
            profile = builder.create_optimization_profile()
            for name, shape in input_shapes.items():
                profile.set_shape(name, shape, shape, shape)
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved: {engine_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting ONNX to TensorRT: {e}")
            return False
    
    def _load_engine(self, engine_path: str, model_name: str) -> bool:
        """Load TensorRT engine"""
        try:
            if not self.trt_available:
                return False
            
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create runtime
            runtime = trt.Runtime(trt_logger)
            
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                logger.error(f"Failed to load TensorRT engine: {engine_path}")
                return False
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Store engine and context
            self.engines[model_name] = engine
            self.contexts[model_name] = context
            
            logger.info(f"TensorRT engine loaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TensorRT engine: {e}")
            return False
    
    def infer(self, model_name: str, inputs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Run inference using TensorRT engine"""
        try:
            if not self.trt_available or model_name not in self.engines:
                logger.warning(f"TensorRT engine not available for {model_name}")
                return None
            
            engine = self.engines[model_name]
            context = self.contexts[model_name]
            
            # Allocate GPU memory
            bindings = []
            outputs = {}
            
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = engine.get_binding_shape(i)
                binding_size = trt.volume(binding_shape) * engine.max_batch_size
                binding_dtype = trt.nptype(engine.get_binding_dtype(i))
                
                # Allocate GPU memory
                gpu_mem = cuda.mem_alloc(binding_size * binding_dtype().itemsize)
                bindings.append(int(gpu_mem))
                
                if engine.binding_is_input(i):
                    # Copy input data to GPU
                    input_data = inputs[binding_name]
                    cuda.memcpy_htod(gpu_mem, input_data)
                else:
                    # Prepare output buffer
                    output_data = np.empty(binding_shape, dtype=binding_dtype)
                    outputs[binding_name] = output_data
            
            # Run inference
            context.execute_v2(bindings=bindings)
            
            # Copy outputs back to CPU
            for i, binding_name in enumerate(engine.get_binding_names()):
                if not engine.binding_is_input(i):
                    cuda.memcpy_dtoh(outputs[binding_name], bindings[i])
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error during TensorRT inference: {e}")
            return None
    
    def benchmark_model(self, model_name: str, input_shapes: Dict[str, Tuple[int, ...]], 
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark TensorRT model performance"""
        try:
            if not self.trt_available or model_name not in self.engines:
                return {"error": "TensorRT engine not available"}
            
            # Create dummy inputs
            inputs = {}
            for name, shape in input_shapes.items():
                inputs[name] = np.random.randn(*shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                self.infer(model_name, inputs)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                self.infer(model_name, inputs)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / num_iterations
            fps = 1.0 / avg_time
            
            return {
                "total_time": total_time,
                "avg_inference_time": avg_time,
                "fps": fps,
                "iterations": num_iterations
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return {"error": f"Benchmark error: {str(e)}"}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded TensorRT model"""
        try:
            if not self.trt_available or model_name not in self.engines:
                return {"error": "TensorRT engine not available"}
            
            engine = self.engines[model_name]
            
            info = {
                "model_name": model_name,
                "num_bindings": engine.num_bindings,
                "max_batch_size": engine.max_batch_size,
                "inputs": {},
                "outputs": {}
            }
            
            for i in range(engine.num_bindings):
                binding_name = engine.get_binding_name(i)
                binding_shape = engine.get_binding_shape(i)
                binding_dtype = str(engine.get_binding_dtype(i))
                
                binding_info = {
                    "shape": binding_shape,
                    "dtype": binding_dtype,
                    "size": trt.volume(binding_shape)
                }
                
                if engine.binding_is_input(i):
                    info["inputs"][binding_name] = binding_info
                else:
                    info["outputs"][binding_name] = binding_info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": f"Info error: {str(e)}"}
    
    def optimize_mediapipe_model(self, model_path: str, model_name: str) -> bool:
        """Optimize MediaPipe model for TensorRT"""
        try:
            # This is a placeholder for MediaPipe model optimization
            # In practice, this would involve converting MediaPipe models to TensorRT
            logger.info(f"MediaPipe model optimization not implemented yet: {model_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error optimizing MediaPipe model: {e}")
            return False
    
    def cleanup(self):
        """Clean up TensorRT resources"""
        try:
            for context in self.contexts.values():
                del context
            
            for engine in self.engines.values():
                del engine
            
            self.contexts.clear()
            self.engines.clear()
            
            logger.info("TensorRT resources cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def is_available(self) -> bool:
        """Check if TensorRT is available"""
        return self.trt_available and torch.cuda.is_available()
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
            
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            cached_memory = torch.cuda.memory_reserved(0)
            
            return {
                "total_memory_gb": total_memory / (1024**3),
                "allocated_memory_gb": allocated_memory / (1024**3),
                "cached_memory_gb": cached_memory / (1024**3),
                "free_memory_gb": (total_memory - allocated_memory) / (1024**3),
                "utilization": allocated_memory / total_memory
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU memory usage: {e}")
            return {"error": f"Memory error: {str(e)}"}
    
    def optimize_for_video_processing(self, input_resolution: Tuple[int, int], 
                                    batch_size: int = 1) -> Dict[str, Any]:
        """Optimize settings for video processing"""
        try:
            width, height = input_resolution
            
            # Calculate optimal batch size based on resolution and available memory
            memory_info = self.get_gpu_memory_usage()
            
            if "error" in memory_info:
                return memory_info
            
            free_memory_gb = memory_info["free_memory_gb"]
            
            # Estimate memory usage per frame (rough calculation)
            bytes_per_pixel = 3  # RGB
            frame_memory_mb = (width * height * bytes_per_pixel) / (1024**2)
            
            # Reserve some memory for model weights and other operations
            available_memory_mb = free_memory_gb * 1024 * 0.7  # Use 70% of free memory
            
            optimal_batch_size = int(available_memory_mb / frame_memory_mb)
            optimal_batch_size = min(max(1, optimal_batch_size), batch_size)
            
            settings = {
                "optimal_batch_size": optimal_batch_size,
                "frame_memory_mb": frame_memory_mb,
                "available_memory_mb": available_memory_mb,
                "recommended_settings": {
                    "use_fp16": self.use_fp16,
                    "max_batch_size": optimal_batch_size
                }
            }
            
            return settings
            
        except Exception as e:
            logger.error(f"Error optimizing for video processing: {e}")
            return {"error": f"Optimization error: {str(e)}"}
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()