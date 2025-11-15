"""
Modified Audio Classification Script (Normal vs. Dysarthria)
Processes the entire audio file without dividing into chunks.

NOTE: The model was trained on 3-second segments (128x128 features). 
      This script will crop the full audio's Mel-Spectrogram to 128x128, 
      meaning only the beginning (~3 seconds) of the audio is analyzed.
"""

import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchvision.models as models

# Configuration
RECORDING_DURATION_SEC = 3 # This is still relevant as the training feature size (128x128) maps to this duration.
N_MELS = 128
FEATURE_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = 'speech_classifier_torgo_only.pth' 


class AudioFeatureExtractor(nn.Module):
    """Extract Mel-Spectrogram features from audio arrays"""
    def __init__(self, n_mels=128, sr=22050):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
    
    def extract_mel_spectrogram_from_array(self, y):
        """Generate Normalized Mel-Spectrogram from a waveform array"""
        try:
            # Generate mel spectrogram from the full array 'y'
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
    """
    Modified ResNet18 for Mel-Spectrogram classification.
    num_classes defaults to 2 for Normal vs. Dysarthria.
    """
    def __init__(self, num_classes=2): 
        super().__init__()
        self.resnet = models.resnet18(weights=None) 
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def classify_audio(audio_path, model_path=DEFAULT_MODEL_PATH):
    """Classify the entire audio file (with fixed input size constraint)"""

    # Validate paths
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print(f"Please train the model first and ensure '{DEFAULT_MODEL_PATH}' exists.")
        return None
    
    # Load model
    print(f"Loading model from: {model_path}")
    # map_location='cpu' is safer for general use, then explicitly moving to DEVICE
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Load from checkpoint with necessary metadata
    if 'model_state_dict' not in checkpoint or 'label_mapping' not in checkpoint:
        print("Error: Invalid model file format. Missing 'model_state_dict' or 'label_mapping'.")
        print("Ensure the file is a checkpoint saved by the training script.")
        return None
        
    label_mapping = checkpoint['label_mapping']
    class_names = checkpoint.get('class_names', [label_mapping[i] for i in range(len(label_mapping))])
    
    # Dynamically create model instance with the correct number of classes
    model = ResNet18Feature(num_classes=len(label_mapping))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    print(f"Classification targets: {class_names}\n")

    feature_extractor = AudioFeatureExtractor(n_mels=N_MELS)

    # 1. Load full audio
    y, sr = librosa.load(audio_path, sr=feature_extractor.sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Processing full audio file (Duration: {duration:.2f}s)")
    
    # 2. Extract Mel-Spectrogram
    mel_spec = feature_extractor.extract_mel_spectrogram_from_array(y)
    
    if mel_spec is None:
        print("Error: Feature extraction failed for the entire audio.")
        return None
        
    # 3. Apply Cropping/Padding to enforce 128x128 input shape
    # This is necessary because the model only accepts 128x128 input.
    # It effectively limits the analysis to the first ~3 seconds of the audio.
    h, w = mel_spec.shape
    print(f"Raw Mel-Spectrogram shape: {mel_spec.shape}. Cropping/Padding to {FEATURE_SIZE}x{FEATURE_SIZE}.")

    # Crop to 128x128
    mel_spec_cropped = mel_spec[:FEATURE_SIZE, :FEATURE_SIZE]

    # Pad if necessary (e.g., if audio is very short)
    pad_h = max(0, FEATURE_SIZE - mel_spec_cropped.shape[0])
    pad_w = max(0, FEATURE_SIZE - mel_spec_cropped.shape[1])
    mel_spec_final = np.pad(mel_spec_cropped, ((0, pad_h), (0, pad_w)), mode='constant')

    # Preprocessing for CNN: (1, 1, 128, 128)
    X = torch.from_numpy(mel_spec_final).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 4. Single Prediction
    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

    # 5. Final Result
    final_idx = int(np.argmax(probabilities))
    final_class = class_names[final_idx]
    final_conf = probabilities[final_idx] * 100

    # Printing final result
    print("\n" + "="*60)
    print("Final Classification (Single Analysis)")
    print("="*60)
    print(f"Predicted Class: {final_class}")
    print(f"Confidence: {final_conf:.2f}%")
    print("\nClass Probability Breakdown:")
    for i, cname in enumerate(class_names):
        print(f"{cname}: {probabilities[i]*100:.2f}%")
    print("="*60 + "\n")

    return final_class, final_conf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_audio.py <audio_file> [model_path]")
        print(f"Default model path: {DEFAULT_MODEL_PATH}")
        sys.exit(1)

    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    classify_audio(audio_path, model_path)