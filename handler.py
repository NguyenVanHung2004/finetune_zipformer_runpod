# -*- coding: utf-8 -*-
"""
handler.py – RunPod Serverless Handler for Zipformer Fine-tuning.

Nhận input:
    - jsonl_url: Link tải file JSONL chứa dữ liệu train (audio_url, transcript)
    - github_token: (Optional) GitHub PAT để upload kết quả
    - github_repo: (Optional) Repo đích (user/repo)
    - tag: (Optional) Tên tag cho GitHub Release (mặc định vYYYY.MM.DD.HHMM)
    - num_epochs: (Optional) Số epoch train (default: 3)
    - adapter_dim: (Optional) Dimension của adapter (default: 8)
    - base_lr: (Optional) Learning rate (default: 0.01)

Trả về:
    - status: 'success' hoặc 'error'
    - release_url: Link GitHub Release nếu có upload
    - download_urls: Danh sách link tải file ONNX
"""

import os
import json
import logging
import requests
import datetime
import runpod
from finetune_core import FinetunePipeline
from upload_github import upload_onnx_to_github

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def download_jsonl(url, dest="/tmp/input.jsonl"):
    """Tải file JSONL từ URL về local."""
    log.info(f"Downloading JSONL from {url}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    # Ghi file, xử lý newline nếu cần
    with open(dest, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return dest

def handler(job):
    """
    Main handler function for RunPod Serverless.
    """
    job_input = job["input"]
    
    # 1. Parse inputs
    jsonl_url    = job_input.get("jsonl_url")
    github_token = job_input.get("github_token")
    github_repo  = job_input.get("github_repo")
    num_epochs   = job_input.get("num_epochs", 3)
    adapter_dim  = job_input.get("adapter_dim", 8)
    base_lr      = job_input.get("base_lr", 0.01)
    
    if not jsonl_url:
        return {"status": "error", "message": "Missing 'jsonl_url' in input."}

    # Tạo tag mặc định nếu không có
    now = datetime.datetime.now().strftime("%Y.%m.%d.%H%M")
    tag = job_input.get("tag", f"v{now}")

    try:
        # 2. Chuẩn bị file JSONL
        jsonl_path = download_jsonl(jsonl_url)
        
        # 3. Khởi tạo và chạy pipeline
        # Network Volume được mount tại /runpod-volume → dung lượng lớn, persistent.
        # Checkpoints (~600MB/epoch), pretrained model (~600MB), audio đều lưu ở đây.
        pipeline = FinetunePipeline(
            base_dir="/runpod-volume/finetune_data",
            num_epochs=num_epochs,
            adapter_dim=adapter_dim,
            base_lr=base_lr
        )
        
        onnx_dir = pipeline.run_all(jsonl_path)
        
        # 4. Upload lên GitHub nếu đủ thông tin
        release_url = None
        download_urls = []
        
        if github_token and github_repo:
            log.info(f"Uploading results to GitHub: {github_repo}...")
            release_url, download_urls = upload_onnx_to_github(
                onnx_dir=onnx_dir,
                repo=github_repo,
                token=github_token,
                tag=tag,
                run_info={
                    "num_epochs": num_epochs,
                    "adapter_dim": adapter_dim,
                    "base_lr": base_lr,
                    "job_id": job["id"]
                }
            )
        else:
            log.warning("GitHub token or repo missing. Skipping upload.")
            # Nếu không upload GitHub, có thể trả về thông tin file trong onnx_dir
            # Nhưng ở Serverless, file trong container sẽ mất khi job xong.
            # Nên KHUYẾN KHÍCH dùng GitHub hoặc các bước upload S3.
        
        return {
            "status": "success",
            "job_id": job["id"],
            "tag": tag,
            "release_url": release_url,
            "download_urls": download_urls
        }

    except Exception as e:
        log.exception("Error during fine-tuning job:")
        return {
            "status": "error",
            "message": str(e),
            "job_id": job["id"]
        }

# Start the serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
