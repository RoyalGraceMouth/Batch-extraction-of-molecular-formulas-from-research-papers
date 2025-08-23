import os
import requests
from tqdm import tqdm

def download_molscribe_model():
    """
    手动下载 MolScribe 模型文件到本地。
    """
    # 模型文件的官方下载链接
    url = "https://github.com/thomas-yan-cheng/MolScribe/releases/download/v0.1.0-alpha/staker_plus_large_20230531.ckpt"
    
    # 定义本地存放目录和文件路径
    model_dir = "models"
    model_path = os.path.join(model_dir, "staker_plus_large_20230531.ckpt")

    # 1. 确保模型目录存在
    os.makedirs(model_dir, exist_ok=True)

    # 2. 如果模型文件已存在，则跳过下载
    if os.path.exists(model_path):
        print(f"模型文件已存在于: {model_path}")
        print("无需下载。")
        return

    # 3. 如果文件不存在，则开始下载
    print(f"模型文件不存在。正在从以下地址下载:")
    print(url)
    
    try:
        # 使用 requests 库进行流式下载，以便显示进度条
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求失败 (如 404)，则会抛出异常

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        with open(model_path, 'wb') as f, tqdm(
            desc="下载中",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        print(f"\n模型下载成功，已保存至: {model_path}")

    except requests.exceptions.RequestException as e:
        print(f"\n下载失败: {e}")
        print("请检查您的网络连接，或尝试手动从上面的URL下载文件，并将其放入 'models' 文件夹中。")
    except Exception as e:
        print(f"\n处理文件时发生未知错误: {e}")


if __name__ == "__main__":
    # 为了运行此脚本，我们需要安装 requests 和 tqdm
    # 请在激活 conda 环境后运行:
    # pip install requests tqdm
    download_molscribe_model()