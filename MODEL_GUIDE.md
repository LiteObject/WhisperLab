# WhisperLab Model Configuration Guide

## What are Whisper Models?

**OpenAI Whisper** is a state-of-the-art automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data. Whisper models are neural networks that convert spoken language into written text with remarkable accuracy.

### Key Features of Whisper Models:

- **🌍 Multilingual Support**: Recognizes speech in 99+ languages
- **🎯 High Accuracy**: State-of-the-art performance on diverse audio
- **🔄 Robust Processing**: Handles accents, background noise, and technical language
- **⚡ Multiple Sizes**: Different model sizes for various speed/accuracy needs
- **🎵 Audio Understanding**: Can handle music, background noise, and poor audio quality

### How Whisper Models Work:

1. **Audio Input**: Takes raw audio (speech, recordings, microphone input)
2. **Feature Extraction**: Converts audio into spectrograms and features
3. **Neural Processing**: Uses transformer architecture to understand speech patterns
4. **Text Output**: Produces accurate transcriptions with punctuation and formatting

### Why Different Model Sizes?

Whisper comes in different sizes because there's always a trade-off between:

- **🚀 Speed vs 🎯 Accuracy**: Smaller models are faster but less accurate
- **💾 Memory vs 📊 Performance**: Larger models need more RAM but work better
- **⚡ Real-time vs 🔍 Batch**: Different use cases need different optimizations

**Real-time transcription** (like WhisperLab) benefits from smaller, faster models, while **post-processing** of important content benefits from larger, more accurate models.

---

## How to Use Different Whisper Models

### Method 1: Configuration File (Recommended)

1. **Use one of the pre-made configuration files:**

   ```bash
   # For better accuracy (slower)
   python run.py --config config-high-accuracy.json
   
   # For maximum speed (lower accuracy)
   python run.py --config config-fast.json
   
   # For English-only content (faster for English)
   python run.py --config config-english-only.json
   ```

2. **Or create your own config file:**
   
   Create a file `my-config.json`:
   ```json
   {
     "whisper": {
       "model_name": "small"
     }
   }
   ```
   
   Then run:
   ```bash
   python run.py --config my-config.json
   ```

### Method 2: Command Line Override

```bash
# Override model directly via command line
python run.py --model tiny     # Fastest
python run.py --model base     # Default
python run.py --model small    # Better accuracy
python run.py --model medium   # High accuracy
python run.py --model large    # Best accuracy

# English-only models (faster for English)
python run.py --model tiny.en
python run.py --model base.en
python run.py --model small.en
```

### Method 3: Environment Variable

```bash
# Windows
set WHISPERLAB_MODEL=small
python run.py

# Linux/Mac
export WHISPERLAB_MODEL=small
python run.py
```

## Model Comparison

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny` | 39MB | ⚡⚡⚡⚡⚡ | ⭐⭐ | Real-time, low-resource |
| `base` | 74MB | ⚡⚡⚡⚡ | ⭐⭐⭐ | **Default choice** |
| `small` | 244MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good balance |
| `medium` | 769MB | ⚡⚡ | ⭐⭐⭐⭐⭐ | High accuracy |
| `large` | 1550MB | ⚡ | ⭐⭐⭐⭐⭐ | Best quality |

## Examples

```bash
# Quick transcription with maximum speed
python run.py --config config-fast.json

# High-quality transcription for important content
python run.py --config config-high-accuracy.json

# English-only content (meetings, interviews)
python run.py --config config-english-only.json

# Quickly test a different model
python run.py --model small
```

## Tips

- **First run**: Models are downloaded automatically and cached
- **Real-time use**: Stick with `tiny`, `base`, or `small`
- **Post-processing**: Use `medium` or `large` for better accuracy
- **English content**: Use `.en` models for better performance
- **Multiple languages**: Use regular models without `.en` suffix