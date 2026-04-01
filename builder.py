# -*- coding: utf-8 -*-
"""
builder.py – Chạy khi BUILD Docker image.

Nhiệm vụ:
  1. Patch icefall/egs/librispeech/ASR/zipformer_adapter/train.py
     để dùng manifest cuts_S / cuts_DEV thay vì GigaSpeech cuts.
  2. Patch asr_datamodule.py để dùng SimpleCutSampler.

Không tải model hay data ở bước này (tiết kiệm image size).
"""

import os
import sys
import subprocess

ICEFALL_DIR = os.environ.get("ICEFALL_DIR", "/icefall")
TRAIN_PY = f"{ICEFALL_DIR}/egs/librispeech/ASR/zipformer_adapter/train.py"
DM_PY    = f"{ICEFALL_DIR}/egs/librispeech/ASR/zipformer_adapter/asr_datamodule.py"


# ─────────────────────────────────────────────────────────────
# Patch 1 – train.py: dùng manifest Vietnamese thay GigaSpeech
# ─────────────────────────────────────────────────────────────
NEW_TRAIN_BLOCK = """\
    from lhotse import load_manifest_lazy as _lml
    gigaspeech_cuts = _lml(params.manifest_dir / "cuts_S.jsonl.gz")

    train_cuts = gigaspeech_cuts
    logging.info(train_cuts)

    def remove_short_and_long_utt(c: Cut):
        if c.duration < 1.0 or c.duration > 20.0:
            return False
        if c.num_frames is None:
            return True
        T = ((c.num_frames - 7) // 2 + 1) // 2
        tokens = params.tokenizer.encode(c.supervisions[0].text, out_type=str)
        if T < len(tokens):
            logging.warning(
                f"Exclude cut ID={c.id}: frames={c.num_frames}, T={T}, "
                f"tokens={len(tokens)}: {c.supervisions[0].text}"
            )
            return False
        return True

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = _lml(params.manifest_dir / "cuts_DEV.jsonl.gz")
    valid_dls = [librispeech.valid_dataloaders(valid_cuts)]
    valid_sets = ["vietnamese"]

"""

NEW_SAMPLER_BLOCK = """\
        valid_sampler = SimpleCutSampler(
            cuts_valid,
            max_cuts=8,
            shuffle=False,
        )
"""


def patch_train_py():
    with open(TRAIN_PY, "r") as f:
        lines = f.readlines()

    if any("cuts_S.jsonl.gz" in l for l in lines):
        print("✅ train.py đã được patch rồi.")
        return

    start_line = end_line = None
    for i, line in enumerate(lines):
        if "gigaspeech_cuts" in line and start_line is None:
            start_line = i
        if start_line is not None and "valid_sets" in line:
            end_line = i + 1
            break

    if start_line is not None and end_line is not None:
        new_lines = lines[:start_line] + [NEW_TRAIN_BLOCK] + lines[end_line:]
        new_lines = [l for l in new_lines if "gigaspeech_dev_cuts" not in l]
        with open(TRAIN_PY, "w") as f:
            f.writelines(new_lines)
        print("✅ Patch train.py OK")
    else:
        print("⚠️  Không tìm thấy block trong train.py – kiểm tra thủ công.")


def patch_datamodule():
    with open(DM_PY, "r") as f:
        dm_lines = f.readlines()

    dm_content = "".join(dm_lines)
    if "DynamicBucketingSampler" not in dm_content:
        print("✅ asr_datamodule.py đã được patch rồi.")
        return

    patched = False
    for i, line in enumerate(dm_lines):
        if "valid_sampler = DynamicBucketingSampler" in line:
            end_i = i + 1
            while end_i < len(dm_lines) and ")" not in dm_lines[end_i]:
                end_i += 1
            end_i += 1
            dm_lines[i:end_i] = [NEW_SAMPLER_BLOCK]
            print(f"  Đã thay thế DynamicBucketingSampler tại dòng {i+1}–{end_i}")
            patched = True
            break

    if not patched:
        # Fallback: tìm rộng hơn
        for i, line in enumerate(dm_lines):
            if "DynamicBucketingSampler" in line and "valid" in "".join(dm_lines[max(0, i-3):i+1]):
                end_i = i + 1
                while end_i < len(dm_lines) and ")" not in dm_lines[end_i]:
                    end_i += 1
                end_i += 1
                dm_lines[i:end_i] = [NEW_SAMPLER_BLOCK]
                print(f"  Đã thay thế tại dòng {i+1}–{end_i} (fallback)")
                patched = True
                break

    if not patched:
        print("❌ Không patch được asr_datamodule.py tự động!")
        return

    # Thêm import SimpleCutSampler nếu chưa có
    dm_content2 = "".join(dm_lines)
    if "SimpleCutSampler" not in dm_content2:
        for i, line in enumerate(dm_lines):
            if "from lhotse.dataset" in line:
                dm_lines.insert(i + 1, "from lhotse.dataset.sampling import SimpleCutSampler\n")
                break

    with open(DM_PY, "w") as f:
        f.writelines(dm_lines)
    print("✅ Patch asr_datamodule.py OK")


def verify_syntax():
    for fp, name in [(TRAIN_PY, "train.py"), (DM_PY, "asr_datamodule.py")]:
        r = subprocess.run([sys.executable, "-m", "py_compile", fp],
                           capture_output=True, text=True)
        status = "✅ OK" if r.returncode == 0 else f"❌ {r.stderr}"
        print(f"  Syntax {name}: {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("BUILDER: Patching icefall scripts")
    print("=" * 60)
    patch_train_py()
    patch_datamodule()
    verify_syntax()
    print("🎉 Builder xong!")
