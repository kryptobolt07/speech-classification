"""
Audio Classification Script
Classify an audio file by splitting it into 3-second chunks and averaging predictions.
"""

import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchvision.models as models

# Configuration
RECORDING_DURATION_SEC = 3
N_MELS = 128
FEATURE_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioFeatureExtractor(nn.Module):
    """Extract Mel-Spectrogram features from audio arrays"""
    def __init__(self, n_mels=128, sr=22050):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
    
    def extract_mel_spectrogram_from_array(self, y):
        """Generate Normalized Mel-Spectrogram from a waveform array"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels, n_fft=2048, hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            if mel_spec_db.max() - mel_spec_db.min() > 0:
                mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            else:
                mel_spec_norm = mel_spec_db
            return mel_spec_norm

        except Exception as e:
            print(f"Error extracting mel-spectrogram: {e}")
            return None


class ResNet18Feature(nn.Module):
    """Modified ResNet18 for Mel-Spectrogram classification"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def classify_audio(audio_path, model_path='speech_classifier_model.pth'):
    """Classify an audio file by splitting it into 3-sec chunks"""

    # Validate paths
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first.")
        return None
    
    # Load model
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # Handle checkpoints
    if 'model_state_dict' in checkpoint:
        label_mapping = checkpoint['label_mapping']
        class_names = checkpoint.get('class_names', [label_mapping[i] for i in range(len(label_mapping))])
        model = ResNet18Feature(num_classes=len(label_mapping))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: Model file is state_dict only.")
        label_mapping = {0: 'Dysarthric', 1: 'Normal', 2: 'Stutter'}
        class_names = ['Dysarthric', 'Normal', 'Stutter']
        model = ResNet18Feature(num_classes=3)
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!\n")

    feature_extractor = AudioFeatureExtractor(n_mels=N_MELS)

    # Load full audio
    y, sr = librosa.load(audio_path, sr=feature_extractor.sr, mono=True)

    chunk_len = RECORDING_DURATION_SEC * sr

    # Pad if shorter
    if len(y) < chunk_len:
        y = np.pad(y, (0, chunk_len - len(y)))
    
    # Create chunks
    chunks = [
        y[i:i + chunk_len]
        for i in range(0, len(y), chunk_len)
        if len(y[i:i + chunk_len]) >= sr
    ]

    print(f"Total chunks detected: {len(chunks)}")

    all_probs = []

    # Process each chunk independently
    for idx, chunk in enumerate(chunks, start=1):
        mel_spec = feature_extractor.extract_mel_spectrogram_from_array(chunk)
        if mel_spec is None:
            continue

        # Resize/pad to 128x128
        h, w = mel_spec.shape
        mel_spec = mel_spec[:FEATURE_SIZE, :FEATURE_SIZE]

        pad_h = max(0, FEATURE_SIZE - mel_spec.shape[0])
        pad_w = max(0, FEATURE_SIZE - mel_spec.shape[1])
        mel_spec = np.pad(mel_spec, ((0, pad_h), (0, pad_w)), mode='constant')

        X = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(X)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        print(f"Chunk {idx} probabilities: {probs}")
        all_probs.append(probs)

    if len(all_probs) == 0:
        print("Error: No valid chunks processed.")
        return None

    # Average probabilities across all chunks
    avg_probs = np.mean(np.vstack(all_probs), axis=0)

    final_idx = int(np.argmax(avg_probs))
    final_class = class_names[final_idx]
    final_conf = avg_probs[final_idx] * 100

    # Printing final result
    print("\n" + "="*60)
    print(f"Final Classification (Averaged over {len(all_probs)} chunks)")
    print("="*60)
    print(f"Predicted Class: {final_class}")
    print(f"Confidence: {final_conf:.2f}%")
    print("\nClass Probability Breakdown:")
    for i, cname in enumerate(class_names):
        print(f"{cname}: {avg_probs[i]*100:.2f}%")
    print("="*60 + "\n")

    return final_class, final_conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_audio.py <audio_file> [model_path]")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'speech_classifier_model.pth'

    classify_audio(audio_path, model_path)
