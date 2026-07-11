# IndicF5 Tamil Fine-Tuning & Voice Cloning

This repository contains the codebase to fine-tune the [IndicF5](https://huggingface.co/ai4bharat/IndicF5) TTS model for high-quality, natural Tamil speech synthesis and zero-shot voice cloning. It uses the `ragunath-ravi/TamilVoiceCorpus` and `ragunath-ravi/TTSvoiceCorpus` datasets.

To maximize vocal naturalness and pronunciation flow without disrupting the pre-trained model weights, this codebase implements **DiT LoRA (Low-Rank Adaptation)** and an offline weight-merging utility.

---

## 🚀 Features

- **DiT LoRA Fine-Tuning**: Injects trainable rank-16 adapters directly into the Attention layers of all 22 backbone blocks, leaving base weights untouched.
- **Offline Weight Merger**: Automatically merges the learned LoRA parameters back into the base weights, outputting a standard F5-TTS checkpoint file compatible with all standard loaders (like Hugging Face `AutoModel`).
- **Zero-Shot Voice Cloning**: Adapt to any target voice instantly using a short (5-15s) reference audio clip and its transcript.

---

## 🛠️ Setup

### Prerequisites
- Python 3.10 to 3.12
- NVIDIA GPU with CUDA (Tesla T4 or better)
- [uv](https://docs.astral.sh/uv/) package manager (optional, pip works too)

### Installation
```bash
# Clone the repository
git clone https://github.com/Ragu-123/indicf5.git
cd indicf5

# Install the package in editable mode
pip install -e .

# Pin numpy to 2.0.0 for compatibility with numba and Kaggle environments
pip install numpy==2.0.0
```

---

## 🎙️ Usage Workflow

### Step 1: Prepare Tamil Dataset
Downloads and resamples the specified subset of the Tamil Voice Corpus to 24kHz:
```bash
python src/indicf5_finetune/prepare_tamil_data.py \
    --data-dir ./data_tamil \
    --subset PureVox \
    --split train
```

### Step 2: Run LoRA Fine-Tuning (Multi-GPU)
Trains the model using LoRA adapters on dual GPUs:
```bash
python -m accelerate.commands.launch --multi_gpu --num_processes 2 --mixed_precision no \
    -m indicf5_finetune.train \
    --data-dir ./data_tamil \
    --use-lora \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 4000 \
    --grad-accum 24 \
    --num-workers 0
```

### Step 3: Merge LoRA Weights
Merge the trained LoRA parameters back into the base weights to get a standard checkpoint:
```bash
python src/indicf5_finetune/merge_lora.py \
    --checkpoint-path ./checkpoints/model_last.pt \
    --output-path ./checkpoints/model_last.pt
```

### Step 4: Generate Audio (Inference)
Clone a voice and synthesize a custom Tamil sentence:
```bash
python -m indicf5_finetune.evaluate \
    --checkpoint-dir ./checkpoints \
    --ref-audio ./ref_audio.wav \
    --ref-text "Reference transcription text here"
```

---

## 👤 Author

- **Ragunath Ravi** ([Ragu-123](https://github.com/Ragu-123))

---

## 📄 License

- Fine-tuning code: MIT
- IndicF5 base model: MIT
- F5-TTS code: MIT
