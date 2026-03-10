import os
import gzip
import shutil
import requests
from tqdm import tqdm

def download_fasttext():
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
    compressed_file = "cc.en.300.bin.gz"
    target_file = "fasttext.bin"

    if os.path.exists(target_file):
        print(f"✅ {target_file} 已经存在，无需下载。")
        return

    # 1. 下载压缩包
    print(f"📡 正在从 Facebook 服务器下载模型 (约 4.2GB)...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(compressed_file, "wb") as f, tqdm(
        desc=compressed_file,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

    # 2. 解压缩
    print(f"📦 正在解压文件，请稍候...")
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(target_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # 3. 清理临时文件
    print(f"🧹 正在清理临时压缩包...")
    os.remove(compressed_file)
    print(f"✨ 成功！模型已就绪: {os.path.abspath(target_file)}")

if __name__ == "__main__":
    download_fasttext()