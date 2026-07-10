"""
Download ragunath-ravi/TamilVoiceCorpus dataset from HuggingFace and convert it
to the F5-TTS CustomDatasetPath format (raw/ + duration.json).

Usage:
    # Via uv script
    uv run prepare-tamil-data --data-dir ./data_tamil

    # Or directly
    uv run python -m indicf5_finetune.prepare_tamil_data --data-dir ./data_tamil
"""

import argparse
import json
import os
import shutil
import sys
import soundfile as sf

TARGET_SR = 24_000
MIN_DURATION = 0.3
MAX_DURATION = 30.0


def print_stats(audio_paths, texts, durations):
    """Print dataset statistics."""
    total_hrs = sum(durations) / 3600
    avg_dur = sum(durations) / len(durations) if durations else 0

    tamil_chars = 0
    english_chars = 0
    for text in texts:
        for c in text:
            if "\u0b80" <= c <= "\u0bff":
                tamil_chars += 1
            elif c.isascii() and c.isalpha():
                english_chars += 1

    mix_ratio = english_chars / max(tamil_chars + english_chars, 1) * 100

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"  Utterances:     {len(audio_paths):,}")
    print(f"  Total duration: {total_hrs:.2f} hours")
    print(f"  Avg duration:   {avg_dur:.2f}s")
    if durations:
        print(f"  Min duration:   {min(durations):.2f}s")
        print(f"  Max duration:   {max(durations):.2f}s")
    print(f"  English chars:  {english_chars:,} ({mix_ratio:.1f}%)")
    print(f"  Tamil chars:    {tamil_chars:,} ({100-mix_ratio:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare TamilVoiceCorpus for IndicF5 fine-tuning")
    parser.add_argument("--data-dir", default="./data_tamil", help="Output directory")
    parser.add_argument("--subset", default="PureVox", choices=["PureVox", "AmbientVox"],
                        help="Dataset subset (PureVox has less noise, AmbientVox has more samples)")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit number of samples (0=all, use 10 for test run)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    audio_out_dir = os.path.join(data_dir, "audio")
    os.makedirs(audio_out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Preparing TamilVoiceCorpus ({args.subset} - {args.split} split)")
    print(f"Output: {data_dir}")
    print(f"{'='*60}\n")

    if args.max_samples > 0:
        print(f"*** TEST MODE: limiting to {args.max_samples} samples ***\n")

    # Load dataset
    print("[1/3] Loading dataset from HuggingFace ...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets package not found. Run: pip install datasets")
        sys.exit(1)

    try:
        import datasets
        dataset = load_dataset("ragunath-ravi/TamilVoiceCorpus", name=args.subset, split=args.split)
        dataset = dataset.cast_column("file_path", datasets.Audio(decode=False))
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    print(f"Loaded {len(dataset)} samples. Processing audios...")

    audio_paths = []
    texts = []
    durations = []
    skipped = 0

    for idx, row in enumerate(dataset):
        if args.max_samples > 0 and len(audio_paths) >= args.max_samples:
            break

        text = row.get("text", "")
        if not text or not text.strip():
            skipped += 1
            continue

        audio_data = row.get("file_path", None)
        if audio_data is None:
            skipped += 1
            continue

        import io
        array, sr = None, None
        try:
            if audio_data.get("bytes") is not None:
                array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            elif audio_data.get("path") is not None:
                array, sr = sf.read(audio_data["path"])
        except Exception as e:
            skipped += 1
            continue

        if array is None or sr is None:
            skipped += 1
            continue

        # Calculate duration
        duration = len(array) / sr
        if duration < MIN_DURATION or duration > MAX_DURATION:
            skipped += 1
            continue

        # Convert to mono
        if array.ndim > 1:
            array = array.mean(axis=1)

        # Resample to 24kHz
        if sr != TARGET_SR:
            try:
                import torch
                import torchaudio
                audio_tensor = torch.from_numpy(array).float()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                audio_tensor = resampler(audio_tensor)
                array = audio_tensor.squeeze(0).numpy()
            except ImportError:
                try:
                    import librosa
                    array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)
                except ImportError:
                    print("ERROR: torchaudio or librosa must be installed for resampling.")
                    sys.exit(1)

        # Save audio file
        file_name = row.get("file_name", f"sample_{idx}.wav")
        # Ensure it has .wav suffix and is a pure filename
        if not file_name.endswith(".wav"):
            file_name = file_name.split(".")[0] + ".wav"
        file_name = os.path.basename(file_name)
        dst_path = os.path.join(audio_out_dir, file_name)

        sf.write(dst_path, array, TARGET_SR)

        audio_paths.append(os.path.abspath(dst_path))
        texts.append(text)
        durations.append(round(len(array) / TARGET_SR, 4))

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx+1}/{len(dataset)} samples... ({len(audio_paths)} kept, {skipped} skipped)")

    print(f"\n[2/3] Saving Arrow dataset ...")
    from datasets import Dataset as HFDataset
    out_dataset = HFDataset.from_dict({
        "audio_path": audio_paths,
        "text": texts,
        "duration": durations,
    })
    out_dataset.save_to_disk(os.path.join(data_dir, "raw"))
    print(f"  Saved arrow dataset to {data_dir}/raw/")

    duration_path = os.path.join(data_dir, "duration.json")
    with open(duration_path, "w") as f:
        json.dump({"duration": durations}, f)
    print(f"  Saved {duration_path}")

    print("\n[3/3] Copying vocabulary ...")
    from huggingface_hub import hf_hub_download
    vocab_path = os.path.join(data_dir, "vocab.txt")
    try:
        vocab_src = hf_hub_download("ai4bharat/IndicF5", filename="checkpoints/vocab.txt")
        shutil.copy2(vocab_src, vocab_path)
        print(f"  Copied vocab.txt ({sum(1 for _ in open(vocab_path, encoding='utf-8'))} entries)")
    except Exception as e:
        print(f"  Warning: HF Hub download failed ({e}), checking local cache...")
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5")
        vocab_src = None
        for root, dirs, files in os.walk(hf_cache):
            if "vocab.txt" in files:
                vocab_src = os.path.join(root, "vocab.txt")
                break
        if vocab_src:
            shutil.copy2(vocab_src, vocab_path)
            print("  Copied vocab.txt from local cache.")
        else:
            print("  ERROR: Could not download or locate vocab.txt")
            sys.exit(1)

    print_stats(audio_paths, texts, durations)

    print(f"\nReady for fine-tuning! Run:")
    print(f"  uv run train --data-dir {data_dir}")


if __name__ == "__main__":
    main()
