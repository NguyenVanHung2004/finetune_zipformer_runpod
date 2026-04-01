# -*- coding: utf-8 -*-
"""
upload_github.py – Upload ONNX model files to a GitHub Release.

Dùng GitHub REST API (requests), không cần gh CLI.
File ONNX và tokens.txt được upload lên Release Assets → URL public.

Cách dùng:
    from upload_github import upload_onnx_to_github
    release_url, urls = upload_onnx_to_github(
        onnx_dir   = "/workspace/finetune_data/output_onnx",
        repo       = "your-user/vi-asr-models",
        token      = "ghp_...",
        tag        = "v2024.04.01.1530",
        run_info   = {"epochs": 3, "samples": 120},
    )
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional


GH_API = "https://api.github.com"
UPLOAD_API = "https://uploads.github.com"


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────

def _headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }


def _ensure_repo_exists(repo: str, token: str):
    """Raise helpful error if repo không tồn tại / không có quyền."""
    url = f"{GH_API}/repos/{repo}"
    r = requests.get(url, headers=_headers(token), timeout=15)
    if r.status_code == 404:
        raise RuntimeError(
            f"Repo '{repo}' không tồn tại hoặc token thiếu quyền 'repo'.\n"
            "  → Tạo repo trên https://github.com/new rồi chạy lại."
        )
    r.raise_for_status()


def _delete_release_by_tag(repo: str, token: str, tag: str):
    """Xóa release và tag git nếu đã tồn tại."""
    # 1. Tìm release theo tag
    r = requests.get(
        f"{GH_API}/repos/{repo}/releases/tags/{tag}",
        headers=_headers(token), timeout=15,
    )
    if r.status_code == 200:
        release_id = r.json()["id"]
        # Xóa release
        requests.delete(
            f"{GH_API}/repos/{repo}/releases/{release_id}",
            headers=_headers(token), timeout=15,
        ).raise_for_status()
        print(f"   🗑️  Đã xóa release cũ (id={release_id})")

    # 2. Xóa git tag
    # 204 = xóa thành công
    # 422 = tag không tồn tại → OK
    # 404 = không tìm thấy ref → OK (empty repo hoặc tag chưa tạo)
    # 409 = Conflict (thường gặp với empty repo) → bỏ qua, GitHub tự dọn
    r2 = requests.delete(
        f"{GH_API}/repos/{repo}/git/refs/tags/{tag}",
        headers=_headers(token), timeout=15,
    )
    if r2.status_code in (204, 404, 409, 422):
        print(f"   🗑️  Git tag '{tag}' đã xử lý (status={r2.status_code})")
    else:
        r2.raise_for_status()


def _create_release(repo: str, token: str, tag: str, body: str) -> dict:
    """Tạo GitHub Release. Nếu tag đã tồn tại → tự xóa rồi tạo lại."""
    url = f"{GH_API}/repos/{repo}/releases"
    payload = {
        "tag_name": tag,
        "name": f"Fine-tuned Zipformer {tag}",
        "body": body,
        "draft": False,
        "prerelease": False,
    }
    r = requests.post(url, json=payload, headers=_headers(token), timeout=30)
    if r.status_code == 422:
        # Tag đã tồn tại → xóa và thử lại
        print(f"   ⚠️  Tag '{tag}' đã tồn tại → đang xóa và tạo lại...")
        _delete_release_by_tag(repo, token, tag)
        r = requests.post(url, json=payload, headers=_headers(token), timeout=30)
    r.raise_for_status()
    return r.json()


def _upload_asset(upload_url: str, token: str, file_path: str) -> str:
    """Upload một file lên Release asset. Trả về browser_download_url."""
    # upload_url dạng: https://uploads.github.com/repos/.../assets{?name,label}
    base  = upload_url.split("{")[0]
    name  = os.path.basename(file_path)
    url   = f"{base}?name={name}"
    size  = os.path.getsize(file_path)

    print(f"    ⬆️  {name}  ({size / 1e6:.1f} MB) ...", end=" ", flush=True)

    headers = {**_headers(token), "Content-Type": "application/octet-stream"}
    with open(file_path, "rb") as f:
        r = requests.post(url, headers=headers, data=f,
                          timeout=600)  # 10 min per file
    r.raise_for_status()
    dl_url = r.json()["browser_download_url"]
    print(f"✅")
    return dl_url


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def upload_onnx_to_github(
    onnx_dir: str,
    repo: str,
    token: str,
    tag: str,
    run_info: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """
    Upload tất cả *.onnx + tokens.txt từ onnx_dir lên GitHub Release.

    Returns:
        (release_url, list_of_download_urls)
    """
    print("=" * 60)
    print(f"UPLOAD: GitHub Release  {repo}@{tag}")
    print("=" * 60)

    # 1. Kiểm tra repo
    _ensure_repo_exists(repo, token)

    # 2. Xác định file cần upload
    onnx_path = Path(onnx_dir)
    files = sorted(onnx_path.glob("*.onnx"))
    tokens_file = onnx_path / "tokens.txt"
    if tokens_file.exists():
        files.append(tokens_file)

    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy file ONNX trong: {onnx_dir}\n"
            "  → Kiểm tra STEP 5 (export_onnx) đã chạy thành công chưa."
        )

    file_list = "\n".join(f"- `{f.name}` ({f.stat().st_size / 1e6:.1f} MB)"
                          for f in files)

    # 3. Tạo release body
    info_str = json.dumps(run_info, ensure_ascii=False, indent=2) if run_info else "{}"
    body = f"""## 🎙️ Vietnamese Zipformer Adapter – {tag}

Auto-generated by RunPod fine-tune pipeline.

### Files
{file_list}

### Training config
```json
{info_str}
```
"""

    # 4. Tạo Release
    release = _create_release(repo, token, tag, body)
    upload_url   = release["upload_url"]
    release_url  = release["html_url"]
    print(f"✅ Release tạo thành công: {release_url}")

    # 5. Upload từng file
    download_urls = []
    for fp in files:
        dl_url = _upload_asset(upload_url, token, str(fp))
        download_urls.append(dl_url)

    # 6. Upload manifest JSON (để server biết tên file cần tải)
    manifest = {
        "tag": tag,
        "repo": repo,
        "release_url": release_url,
        "files": [os.path.basename(u) for u in download_urls],
        "download_urls": download_urls,
        **(run_info or {}),
    }
    manifest_path = "/tmp/model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    manifest_url = _upload_asset(upload_url, token, manifest_path)
    download_urls.append(manifest_url)

    print(f"\n🎉 Upload hoàn thành!")
    print(f"   Release:  {release_url}")
    print(f"   Files: {len(download_urls)}")
    return release_url, download_urls
