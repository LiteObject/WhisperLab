# WhisperLab Model Configuration Guide

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