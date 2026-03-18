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
        print(f"✅ {target_file} already exists, skip download.")
        return

    print(f"📡 Downloading model from Facebook (~4.2GB)...")
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

    print(f"📦 Extracting archive...")
    with gzip.open(compressed_file, 'rb') as f_in:
        with open(target_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"🧹 Removing temporary archive...")
    os.remove(compressed_file)
    print(f"✨ Done. Model ready: {os.path.abspath(target_file)}")

if __name__ == "__main__":
    download_fasttext()