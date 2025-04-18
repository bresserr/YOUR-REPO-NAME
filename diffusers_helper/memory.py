# By lllyasviel


import torch
import os

# 检查是否在Hugging Face Space环境中
IN_HF_SPACE = os.environ.get('SPACE_ID') is not None

# 设置CPU设备
cpu = torch.device('cpu')

# 在Stateless GPU环境中，不要在主进程初始化CUDA
def get_gpu_device():
    if IN_HF_SPACE:
        # 在Spaces中将延迟初始化GPU设备
        return 'cuda'  # 返回字符串，而不是实际初始化设备
    
    # 非Spaces环境正常初始化
    try:
        if torch.cuda.is_available():
            return torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            print("CUDA不可用，使用CPU作为默认设备")
            return torch.device('cpu')
    except Exception as e:
        print(f"初始化CUDA设备时出错: {e}")
        print("回退到CPU设备")
        return torch.device('cpu')

# 保存一个字符串表示，而不是实际的设备对象
gpu = get_gpu_device()

gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device):
    # 转换字符串设备为torch.device
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
        
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu
    
    # 如果是字符串，转换为设备
    if isinstance(device, str):
        device = torch.device(device)
    
    # 如果不是CUDA设备，返回默认值
    if device.type != 'cuda':
        print("无法获取非CUDA设备的内存信息，返回默认值")
        return 6.0  # 返回一个默认值
    
    try:
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_inactive_reserved = bytes_reserved - bytes_active
        bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
        return bytes_total_available / (1024 ** 3)
    except Exception as e:
        print(f"获取CUDA内存信息时出错: {e}")
        return 6.0  # 返回一个默认值


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    # 如果是字符串，转换为设备
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    
    # 如果gpu是字符串，转换为设备
    gpu_device = gpu
    if isinstance(gpu_device, str):
        gpu_device = torch.device(gpu_device)

    # 如果目标设备是CPU或当前在CPU上，直接移动
    if target_device.type == 'cpu' or gpu_device.type == 'cpu':
        model.to(device=target_device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            torch.cuda.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    # 如果是字符串，转换为设备
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
    
    # 如果gpu是字符串，转换为设备
    gpu_device = gpu
    if isinstance(gpu_device, str):
        gpu_device = torch.device(gpu_device)

    # 如果目标设备是CPU或当前在CPU上，直接处理
    if target_device.type == 'cpu' or gpu_device.type == 'cpu':
        model.to(device=cpu)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            torch.cuda.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    torch.cuda.empty_cache()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return


def load_model_as_complete(model, target_device, unload=True):
    # 如果是字符串，转换为设备
    if isinstance(target_device, str):
        target_device = torch.device(target_device)
        
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
