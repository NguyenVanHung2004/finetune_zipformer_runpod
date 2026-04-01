# -*- coding: utf-8 -*-
"""
finetune_core.py – Pipeline fine-tune Zipformer (class-based).

Thiết kế cho môi trường Docker trên RunPod:
  - Không có step1_install (tất cả đã pre-installed trong Docker image)
  - Nhận config qua constructor (num_epochs, adapter_dim, v.v.)
  - JSONL path truyền vào step2
  - /workspace là network volume của RunPod (dữ liệu persistent)
  - /icefall là thư mục icefall đã clone + patch lúc build image

Các bước:
  step2  → tải audio từ URL, chuẩn hóa 16kHz WAV
  step3  → tạo Lhotse manifests (90/10 train/val)
  step35 → tải pretrained model từ HuggingFace
  step37 → patch icefall scripts (safe to re-run nếu chưa patch)
  step4  → chạy fine-tune (subprocess → train.py)
  step5  → export ONNX (subprocess → export-onnx.py)
  run_all → chạy toàn bộ theo thứ tự
"""

import glob
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Pretrained model source ──────────────────────────────────────
HF_BASE = "https://huggingface.co/zzasdf/viet_iter3_pseudo_label/resolve/main"


# ─────────────────────────────────────────────────────────────────
class FinetunePipeline:
    """
    Full fine-tune pipeline for Vietnamese Zipformer.

    Args:
        base_dir     : Root dir for data, model, output  (default /workspace/finetune_data)
        icefall_dir  : Pre-cloned icefall dir            (default /icefall)
        num_epochs   : Training epochs                   (default 3)
        adapter_dim  : Adapter bottleneck dim            (default 8)
        base_lr      : Learning rate                     (default 0.01)
    """

    def __init__(
        self,
        base_dir:    str = "/runpod-volume/finetune_data",
        icefall_dir: str = "/icefall",
        num_epochs:  int = 3,
        adapter_dim: int = 8,
        base_lr:   float = 0.01,
    ):
        self.base_dir    = base_dir
        self.icefall_dir = icefall_dir
        self.num_epochs  = num_epochs
        self.adapter_dim = adapter_dim
        self.base_lr     = base_lr

        # Sub-directories (tất cả lưu trên Network Volume)
        self.audio_dir  = os.path.join(base_dir, "am_thanh")
        self.text_dir   = os.path.join(base_dir, "van_ban")
        self.model_dir  = os.path.join(base_dir, "model_vi")
        self.output_dir = os.path.join(base_dir, "output_adapter")
        self.onnx_dir   = os.path.join(base_dir, "output_onnx")
        self.manif_dir  = os.path.join(base_dir, "manifests")
        self.wav_dir    = os.path.join(base_dir, "data_ready", "wav")

        for d in [
            self.audio_dir, self.text_dir, self.model_dir,
            self.output_dir, self.onnx_dir, self.manif_dir, self.wav_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # STEP 2: Download & normalize audio from JSONL
    # ─────────────────────────────────────────────────────────
    def step2_prepare_data(self, jsonl_path: str) -> int:
        """Tải audio từ URL trong JSONL, chuẩn hóa 16kHz WAV.

        Returns: số mẫu hợp lệ (created + skipped).
        """
        print("=" * 60)
        print("STEP 2: TẢI & CHUẨN HÓA DỮ LIỆU")
        print("=" * 60)

        import requests
        import librosa
        import soundfile as sf

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL không tìm thấy: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            raw = f.read().replace("\\n", "\n")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        valid_data = []
        for i, line in enumerate(lines):
            try:
                item = json.loads(line)
                if item.get("audio_url") and item.get("transcript"):
                    valid_data.append(item)
            except json.JSONDecodeError as e:
                print(f"  ❌ Dòng {i+1}: {e}")

        print(f"📋 Dữ liệu hợp lệ: {len(valid_data)} mẫu")
        if not valid_data:
            raise ValueError("JSONL không chứa mẫu hợp lệ nào!")

        created = skipped = errors = 0
        for i, data in enumerate(valid_data):
            file_id   = (
                data.get("id", f"item_{i+1}")
                .replace("/", "_").replace(":", "_")
            )
            audio_dst = os.path.join(self.audio_dir, f"{file_id}.wav")
            text_dst  = os.path.join(self.text_dir,  f"{file_id}.txt")

            if os.path.exists(audio_dst) and os.path.exists(text_dst):
                skipped += 1
                continue

            print(f"  [{i+1}/{len(valid_data)}] {file_id}...", end=" ", flush=True)
            try:
                resp = requests.get(data["audio_url"], timeout=30)
                resp.raise_for_status()
                tmp = f"/tmp/{file_id}_raw"
                with open(tmp, "wb") as tf:
                    tf.write(resp.content)
                y, sr = librosa.load(tmp, sr=16000, mono=True)
                sf.write(audio_dst, y, sr)
                with open(text_dst, "w", encoding="utf-8") as tf:
                    tf.write(data["transcript"].strip())
                if os.path.exists(tmp):
                    os.remove(tmp)
                print("✅")
                created += 1
            except Exception as e:
                print(f"❌ {e}")
                errors += 1

        print(f"\n✅ Mới tải: {created} | Đã có sẵn: {skipped} | Lỗi: {errors}")
        return created + skipped

    # ─────────────────────────────────────────────────────────
    # STEP 3: Lhotse manifests (90/10 split)
    # ─────────────────────────────────────────────────────────
    def step3_build_manifests(self):
        """Tạo Lhotse manifests, chia 90% train / 10% val."""
        print("=" * 60)
        print("STEP 3: TẠO LHOTSE MANIFESTS")
        print("=" * 60)

        import lhotse
        wav_path = Path(self.wav_dir)

        # Copy WAV vào wav_dir nếu chưa có
        for fn in os.listdir(self.audio_dir):
            if fn.endswith(".wav"):
                dst = wav_path / fn
                if not dst.exists():
                    shutil.copy2(os.path.join(self.audio_dir, fn), dst)
        n_wav = len(list(wav_path.glob("*.wav")))
        print(f"✅ WAV files: {n_wav}")

        recordings   = lhotse.RecordingSet.from_dir(
            self.wav_dir, pattern="*.wav", num_jobs=1
        )
        supervisions = []
        for rec in recordings:
            txt = os.path.join(self.text_dir, rec.id + ".txt")
            if os.path.exists(txt):
                with open(txt, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                supervisions.append(lhotse.SupervisionSegment(
                    id=rec.id, recording_id=rec.id,
                    start=0.0, duration=rec.duration,
                    channel=0, text=text, language="vi",
                ))

        print(f"✅ Recordings: {len(recordings)} | Supervisions: {len(supervisions)}")
        if not supervisions:
            raise ValueError("Không có supervision! Kiểm tra lại audio/text.")

        sup_set  = lhotse.SupervisionSet.from_segments(supervisions)
        cuts_all = lhotse.CutSet.from_manifests(
            recordings=recordings, supervisions=sup_set
        )

        all_cuts = list(cuts_all)
        random.seed(42)
        random.shuffle(all_cuts)
        n_val   = max(1, int(len(all_cuts) * 0.10))
        n_train = len(all_cuts) - n_val

        train_cuts = lhotse.CutSet.from_cuts(all_cuts[:n_train])
        val_cuts   = lhotse.CutSet.from_cuts(all_cuts[n_train:])
        print(f"✅ Split: {n_train} Train | {n_val} Val")

        mp = Path(self.manif_dir)
        train_cuts.to_file(str(mp / "cuts_S.jsonl.gz"))
        val_cuts.to_file(str(mp / "cuts_DEV.jsonl.gz"))
        val_cuts.to_file(str(mp / "cuts_TEST.jsonl.gz"))
        cuts_all.to_file(str(mp / "cuts.jsonl.gz"))
        recordings.to_file(str(mp / "recordings.jsonl.gz"))
        sup_set.to_file(str(mp / "supervisions.jsonl.gz"))
        print(f"✅ Manifests: {sorted(os.listdir(self.manif_dir))}")

    # ─────────────────────────────────────────────────────────
    # STEP 3.5: Download pretrained model from HuggingFace
    # ─────────────────────────────────────────────────────────
    def step35_download_model(self):
        """Tải pretrained.pt, tokens.txt, bpe.model từ HuggingFace."""
        print("=" * 60)
        print("STEP 3.5: TẢI PRETRAINED MODEL")
        print("=" * 60)

        import requests

        pretrained_pt = os.path.join(self.model_dir, "pretrained.pt")
        tokens_txt    = os.path.join(self.model_dir, "tokens.txt")
        bpe_dst       = os.path.join(self.model_dir, "bpe.model")

        def _download(url, dst, desc, min_size=1000):
            if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
                print(f"✅ {desc} đã có ({os.path.getsize(dst)/1e6:.1f} MB)")
                return
            print(f"⏳ Tải {desc}...")
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done  = 0
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done/total*100:.1f}% ({done/1e6:.1f}/{total/1e6:.1f} MB)",
                                  end="", flush=True)
            print(f"\n✅ {desc} ({os.path.getsize(dst)/1e6:.1f} MB)")

        _download(f"{HF_BASE}/exp/pretrained.pt",
                  pretrained_pt, "pretrained.pt", min_size=int(1e6))
        _download(f"{HF_BASE}/data/Vietnam_bpe_2000_new/tokens.txt",
                  tokens_txt, "tokens.txt", min_size=100)
        _download(f"{HF_BASE}/data/Vietnam_bpe_2000_new/bpe.model",
                  bpe_dst, "bpe.model", min_size=1000)

        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(bpe_dst)
        print(f"✅ SentencePiece vocab = {sp.GetPieceSize()}")

    # ─────────────────────────────────────────────────────────
    # STEP 3.7: Patch icefall (safe to re-run)
    # ─────────────────────────────────────────────────────────
    def step37_patch_icefall(self):
        """
        Gọi builder.py để patch icefall.  
        builder.py đã chạy lúc build Docker, nên thường skip ngay.
        """
        print("=" * 60)
        print("STEP 3.7: PATCH ICEFALL (verify)")
        print("=" * 60)

        # Chạy builder.py bên trong container (idempotent)
        builder = os.path.join("/app", "builder.py")
        r = subprocess.run([sys.executable, builder], capture_output=False)
        if r.returncode != 0:
            raise RuntimeError("builder.py thất bại!")

    # ─────────────────────────────────────────────────────────
    # STEP 4: Fine-tune
    # ─────────────────────────────────────────────────────────
    def step4_finetune(self):
        """Chạy fine-tune qua train.py của icefall."""
        print("=" * 60)
        print("STEP 4: FINE-TUNE")
        print("=" * 60)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng trên pod này!")

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        # RTX 4090 = 24GB → max_duration=300; nhỏ hơn 12GB → 120
        if vram_gb >= 20:
            max_dur = "300"
        elif vram_gb >= 12:
            max_dur = "200"
        else:
            max_dur = "120"
        print(f"   max-duration: {max_dur}")

        cmd = [
            sys.executable,
            "egs/librispeech/ASR/zipformer_adapter/train.py",
            "--world-size",           "1",
            "--num-epochs",           str(self.num_epochs),
            "--start-epoch",          "1",
            "--exp-dir",              self.output_dir,
            "--bpe-model",            os.path.join(self.model_dir, "bpe.model"),
            "--manifest-dir",         self.manif_dir,
            "--do-finetune",          "True",
            "--finetune-ckpt",        os.path.join(self.model_dir, "pretrained.pt"),
            "--use-adapters",         "True",
            "--adapter-dim",          str(self.adapter_dim),
            "--use-fp16",             "1",
            "--base-lr",              str(self.base_lr),
            "--max-duration",         max_dur,
            "--on-the-fly-feats",     "True",
            "--enable-musan",         "False",
            "--num-buckets",          "1",
            "--bucketing-sampler",    "False",
            "--keep-last-k",          "1",   # Chỉ giữ checkpoint cuối → tiết kiệm disk
            "--num-encoder-layers",   "2,2,3,4,3,2",
            "--downsampling-factor",  "1,2,4,8,4,2",
            "--feedforward-dim",      "512,768,1024,1536,1024,768",
            "--num-heads",            "4,4,4,8,4,4",
            "--encoder-dim",          "192,256,384,512,384,256",
            "--query-head-dim",       "32",
            "--value-head-dim",       "12",
            "--pos-head-dim",         "4",
            "--pos-dim",              "48",
            "--encoder-unmasked-dim", "192,192,256,192,256,192",
            "--cnn-module-kernel",    "31,31,15,15,15,31",
            "--decoder-dim",          "512",
            "--joiner-dim",           "512",
        ]

        print("\n🚀 Bắt đầu fine-tune...\n")
        proc = subprocess.Popen(
            cmd, cwd=self.icefall_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"train.py thất bại (exit {proc.returncode})")
        print("\n✅ Fine-tune hoàn thành!")

    # ─────────────────────────────────────────────────────────
    # STEP 5: Export ONNX
    # ─────────────────────────────────────────────────────────
    def step5_export_onnx(self):
        """Export checkpoint cuối cùng sang ONNX."""
        print("=" * 60)
        print("STEP 5: EXPORT SANG ONNX")
        print("=" * 60)

        checkpoints = sorted(
            glob.glob(os.path.join(self.output_dir, "epoch-*.pt")),
            key=lambda x: int(os.path.basename(x).split("-")[1].split(".")[0]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint trong {self.output_dir}!"
            )

        last_epoch = int(
            os.path.basename(checkpoints[-1]).split("-")[1].split(".")[0]
        )
        print(f"✅ Epoch cuối: {last_epoch}")

        cmd = [
            sys.executable,
            "egs/librispeech/ASR/zipformer_adapter/export-onnx.py",
            "--tokens",              os.path.join(self.model_dir, "tokens.txt"),
            "--use-averaged-model",  "0",
            "--epoch",               str(last_epoch),
            "--avg",                 "1",
            "--exp-dir",             self.output_dir,
            "--use-adapters",        "True",
            "--adapter-dim",         str(self.adapter_dim),
            "--num-encoder-layers",  "2,2,3,4,3,2",
            "--downsampling-factor", "1,2,4,8,4,2",
            "--feedforward-dim",     "512,768,1024,1536,1024,768",
            "--num-heads",           "4,4,4,8,4,4",
            "--encoder-dim",         "192,256,384,512,384,256",
            "--query-head-dim",      "32",
            "--value-head-dim",      "12",
            "--pos-head-dim",        "4",
            "--pos-dim",             "48",
            "--encoder-unmasked-dim","192,192,256,192,256,192",
            "--cnn-module-kernel",   "31,31,15,15,15,31",
            "--decoder-dim",         "512",
            "--joiner-dim",          "512",
            "--causal",              "False",
            "--chunk-size",          "16,32,64,-1",
            "--left-context-frames", "64,128,256,-1",
        ]

        print(f"\n🚀 Export epoch {last_epoch}...\n")
        proc = subprocess.Popen(
            cmd, cwd=self.icefall_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"export-onnx.py thất bại (exit {proc.returncode})")

        # Copy ONNX + tokens sang onnx_dir
        for fp in glob.glob(os.path.join(self.output_dir, "*.onnx")):
            shutil.copy2(fp, self.onnx_dir)
            print(f"📦 {os.path.basename(fp)} → output_onnx/")
        shutil.copy2(os.path.join(self.model_dir, "tokens.txt"), self.onnx_dir)
        print("📦 tokens.txt → output_onnx/")

        # Validate ONNX
        try:
            import onnx
            print("\n--- Kiểm tra ONNX ---")
            for fn in os.listdir(self.onnx_dir):
                if fn.endswith(".onnx"):
                    path = os.path.join(self.onnx_dir, fn)
                    size = os.path.getsize(path) / 1e6
                    try:
                        m = onnx.load(path)
                        onnx.checker.check_model(m)
                        print(f"✅ {fn} ({size:.1f} MB) – OK")
                    except Exception as e:
                        print(f"❌ {fn} ({size:.1f} MB) – LỖI: {e}")
        except ImportError:
            print("⚠️  onnx không cài – bỏ qua validation")

        print(f"\n🎉 ONNX files tại: {self.onnx_dir}")
        return self.onnx_dir

    # ─────────────────────────────────────────────────────────
    # run_all: chạy toàn bộ pipeline
    # ─────────────────────────────────────────────────────────
    def run_all(self, jsonl_path: str) -> str:
        """
        Chạy toàn bộ pipeline.

        Args:
            jsonl_path: đường dẫn file JSONL đã tải về /tmp/

        Returns:
            onnx_dir: thư mục chứa file ONNX xuất ra
        """
        print("\n" + "=" * 60)
        print("🚀 Fine-Tune Pipeline – Zipformer Vietnamese Adapter")
        print("=" * 60)
        print(f"   base_dir    : {self.base_dir}")
        print(f"   icefall_dir : {self.icefall_dir}")
        print(f"   num_epochs  : {self.num_epochs}")
        print(f"   adapter_dim : {self.adapter_dim}")
        print(f"   base_lr     : {self.base_lr}")
        print(f"   jsonl_path  : {jsonl_path}")
        print()

        n = self.step2_prepare_data(jsonl_path)
        if n == 0:
            raise RuntimeError("Không có sample nào được chuẩn bị!")

        self.step3_build_manifests()
        self.step35_download_model()
        self.step37_patch_icefall()
        self.step4_finetune()
        return self.step5_export_onnx()
