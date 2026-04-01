# 🎙️ Zipformer Vietnamese Fine-Tune Pipeline (RunPod)

Pipeline tự động fine-tune mô hình ASR **Zipformer** tiếng Việt trên **RunPod Serverless**, sử dụng adapter-based fine-tuning và export sang ONNX.

---

## Tổng quan

```
JSONL (audio_url + transcript)
        ↓
   [RunPod Job]
        ↓
  1. Tải & chuẩn hóa audio (16kHz WAV)
  2. Tạo Lhotse manifests (90/10 train/val)
  3. Tải pretrained model từ HuggingFace
  4. Fine-tune với Adapter (icefall/train.py)
  5. Export sang ONNX
        ↓
  Upload lên GitHub Releases
```

**Pretrained model:** [`zzasdf/viet_iter3_pseudo_label`](https://huggingface.co/zzasdf/viet_iter3_pseudo_label)  
**Framework:** [icefall](https://github.com/k2-fsa/icefall) + [lhotse](https://github.com/lhotse-speech/lhotse)  
**Model output:** [`NguyenVanHung2004/zipFormerModel`](https://github.com/NguyenVanHung2004/zipFormerModel/releases)

---

## Cấu trúc

```
finetune_runpod/
├── handler.py          # RunPod Serverless entrypoint
├── finetune_core.py    # Pipeline chính (step2 → step5)
├── upload_github.py    # Upload ONNX lên GitHub Releases
├── builder.py          # Patch icefall (chạy lúc build Docker)
├── Dockerfile          # Docker image cho RunPod
└── requirements.txt
```

---

## Input (RunPod Job)

```json
{
  "jsonl_url": "https://example.com/data.jsonl",
  "github_token": "ghp_...",
  "github_repo": "NguyenVanHung2004/zipFormerModel",
  "num_epochs": 3,
  "adapter_dim": 8,
  "base_lr": 0.01,
  "tag": "v2024.01.01.1200"
}
```

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `jsonl_url` | *(bắt buộc)* | URL tải file JSONL chứa dữ liệu train |
| `github_token` | — | GitHub PAT để upload kết quả |
| `github_repo` | — | Repo đích (`user/repo`) |
| `num_epochs` | `3` | Số epoch train |
| `adapter_dim` | `8` | Bottleneck dimension của adapter |
| `base_lr` | `0.01` | Learning rate |
| `tag` | `vYYYY.MM.DD.HHMM` | Tag cho GitHub Release |

### Định dạng JSONL

Mỗi dòng là 1 JSON object:

```json
{"id": "sample_001", "audio_url": "https://...", "transcript": "xin chào"}
{"id": "sample_002", "audio_url": "https://...", "transcript": "cảm ơn bạn"}
```

---

## Output

```json
{
  "status": "success",
  "tag": "v2026.04.01.1530",
  "release_url": "https://github.com/NguyenVanHung2004/zipFormerModel/releases/tag/v...",
  "download_urls": [
    "https://github.com/.../encoder-epoch-3-avg-1.onnx",
    "https://github.com/.../decoder-epoch-3-avg-1.onnx",
    "https://github.com/.../joiner-epoch-3-avg-1.onnx",
    "https://github.com/.../tokens.txt"
  ]
}
```

---

## Deploy lên RunPod

### 1. Build & push Docker image

```bash
docker build -t your-dockerhub/zipformer-finetune:latest .
docker push your-dockerhub/zipformer-finetune:latest
```

### 2. Tạo RunPod Serverless Endpoint

- **Container Image:** `your-dockerhub/zipformer-finetune:latest`
- **Network Volume:** Mount tại `/runpod-volume` (≥ 20GB khuyến nghị)
- **GPU:** RTX 4090 hoặc tương đương

### 3. Gửi job

```python
import runpod

runpod.api_key = "your_runpod_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

result = endpoint.run_sync({
    "jsonl_url": "https://example.com/data.jsonl",
    "github_token": "ghp_...",
    "github_repo": "NguyenVanHung2004/zipFormerModel",
    "num_epochs": 3,
})
print(result)
```

---

## Lưu ý

- **Network Volume** (`/runpod-volume`) lưu toàn bộ data, checkpoint và model → không lo hết disk
- Pretrained model (~600MB) được cache trên Network Volume → lần 2 chạy nhanh hơn
- Chỉ giữ 1 checkpoint tại mỗi thời điểm (`--keep-last-k 1`) → tiết kiệm dung lượng
- GitHub token cần quyền **`repo`** (full) để tạo/xóa Release
