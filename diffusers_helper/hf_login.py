import os
from huggingface_hub import login

def login():
    # 如果是在Hugging Face Space环境中运行，使用环境变量中的token
    if os.environ.get('SPACE_ID') is not None:
        print("Running in Hugging Face Space, using environment HF_TOKEN")
        # Space自带访问权限，无需额外登录
        return

    # 如果本地环境有token，则使用它登录
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("Logging in with HF_TOKEN from environment")
        login(token=hf_token)
        return
        
    # 检查缓存的token
    cache_file = os.path.expanduser('~/.huggingface/token')
    if os.path.exists(cache_file):
        print("Found cached Hugging Face token")
        return

    print("No Hugging Face token found. Using public access.")
    # 无token时使用公共访问，速度可能较慢且有限制
