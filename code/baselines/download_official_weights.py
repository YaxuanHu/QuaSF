"""Download official pretrained weights for Chainsaw and Merizo with resume."""
import os
import sys
import time
import requests


def download(url, out_path, chunk_size=8192, timeout=60):
    """Download with resume and retry."""
    headers = {}
    start = 0
    if os.path.exists(out_path):
        start = os.path.getsize(out_path)
        headers['Range'] = f'bytes={start}-'

    for attempt in range(10):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=timeout)
            if r.status_code == 416:  # Range not satisfiable, already complete
                print(f"Already complete: {out_path} ({start} bytes)")
                return
            if r.status_code not in (200, 206):
                print(f"HTTP {r.status_code} for {url}")
                time.sleep(5)
                continue

            mode = 'ab' if r.status_code == 206 else 'wb'
            total = start + int(r.headers.get('content-length', 0))
            downloaded = start
            with open(out_path, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            print(f"Downloaded: {out_path} ({downloaded} bytes)")
            return
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(5)
            # Update start to current file size for resume
            if os.path.exists(out_path):
                start = os.path.getsize(out_path)
                headers['Range'] = f'bytes={start}-'
    raise RuntimeError(f"Failed to download {url}")


def main():
    base = '/geniusland/home/huyaxuan/code/region_function_old/code/baselines'

    # Chainsaw
    chainsaw_dir = os.path.join(base, 'chainsaw-main', 'saved_models', 'model_v3')
    os.makedirs(chainsaw_dir, exist_ok=True)
    for f in ['weights.pt', 'config.json', 'feature_config.json']:
        url = f'https://raw.githubusercontent.com/JudeWells/chainsaw/main/saved_models/model_v3/{f}'
        out = os.path.join(chainsaw_dir, f)
        print(f"Downloading {f}...")
        download(url, out)

    # Merizo
    merizo_dir = os.path.join(base, 'merizo', 'weights')
    os.makedirs(merizo_dir, exist_ok=True)
    for i in range(3):
        f = f'weights_part_{i}.pt'
        url = f'https://github.com/psipred/Merizo/raw/main/weights/{f}'
        out = os.path.join(merizo_dir, f)
        print(f"Downloading {f}...")
        download(url, out)


if __name__ == '__main__':
    main()
