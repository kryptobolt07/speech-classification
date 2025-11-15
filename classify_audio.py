"""
Audio Classification Script
Classify an audio file by splitting it into 3-second chunks and averaging predictions.
Supports test mode for accuracy evaluation on Tests folder.
"""
import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

# Configuration
RECORDING_DURATION_SEC = 3
N_MELS = 128
FEATURE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path Configuration
MODEL_PATH = './best_model.pth'
TEST_DIR = './Tests'

# Label mapping for test folders
FOLDER_TO_LABEL = {
    'Female_dysarthia_control': 'Normal',
    'Male_dysarthia_control': 'Normal',
    'Female_dysarthia': 'Dysarthric',
    'Male_dysarthia': 'Dysarthric',
    'Stuttering': 'Stutter'
}


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


def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None, None, None
    
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
    
    return model, class_names, label_mapping


def classify_audio(audio_path, model, class_names, verbose=True):
    """Classify an audio file using the entire audio"""
    
    # Validate paths
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return None, None
    
    feature_extractor = AudioFeatureExtractor(n_mels=N_MELS)
    
    # Load full audio
    y, sr = librosa.load(audio_path, sr=feature_extractor.sr, mono=True)
    
    if verbose:
        print(f"Audio duration: {len(y)/sr:.2f} seconds")
    
    # Extract mel-spectrogram from entire audio
    mel_spec = feature_extractor.extract_mel_spectrogram_from_array(y)
    if mel_spec is None:
        print("Error: Failed to extract mel-spectrogram.")
        return None, None
    
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
    
    final_idx = int(np.argmax(probs))
    final_class = class_names[final_idx]
    final_conf = probs[final_idx] * 100
    
    # Printing final result
    if verbose:
        print("\n" + "="*60)
        print("Final Classification")
        print("="*60)
        print(f"Predicted Class: {final_class}")
        print(f"Confidence: {final_conf:.2f}%")
        print("\nClass Probability Breakdown:")
        for i, cname in enumerate(class_names):
            print(f"{cname}: {probs[i]*100:.2f}%")
        print("="*60 + "\n")
    
    return final_class, final_conf


def get_true_label_from_path(file_path):
    """Extract true label from file path based on parent folder"""
    path_parts = Path(file_path).parts
    
    for folder in FOLDER_TO_LABEL.keys():
        if folder in path_parts:
            return FOLDER_TO_LABEL[folder]
    
    return None


def test_accuracy(model, class_names, test_dir=TEST_DIR):
    """Test model accuracy on all files in Tests folder"""
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    print("\n" + "="*70)
    print("RUNNING ACCURACY TEST ON TEST DATASET")
    print("="*70 + "\n")
    
    # Find all .wav files recursively
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                true_label = get_true_label_from_path(full_path)
                if true_label:
                    test_files.append((full_path, true_label))
    
    if len(test_files) == 0:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test files\n")
    
    # Track results
    correct = 0
    total = 0
    confusion = {label: {pred: 0 for pred in class_names} for label in class_names}
    
    # Process each file
    for idx, (file_path, true_label) in enumerate(test_files, 1):
        file_name = os.path.basename(file_path)
        print(f"[{idx}/{len(test_files)}] Testing: {file_name}")
        print(f"True Label: {true_label}")
        
        pred_class, pred_conf = classify_audio(file_path, model, class_names, verbose=False)
        
        if pred_class is None:
            print(f"Skipping file (classification failed)\n")
            continue
        
        print(f"Predicted: {pred_class} ({pred_conf:.2f}%)")
        
        total += 1
        if pred_class == true_label:
            correct += 1
            print("✓ CORRECT\n")
        else:
            print("✗ INCORRECT\n")
        
        confusion[true_label][pred_class] += 1
    
    # Print results
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Total Files: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    
    # Print confusion matrix
    header = "True \\ Pred".ljust(15) + "".join([c.ljust(15) for c in class_names])
    print(header)
    print("-"*70)
    
    for true_label in class_names:
        row = true_label.ljust(15)
        for pred_label in class_names:
            count = confusion[true_label][pred_label]
            row += str(count).ljust(15)
        print(row)
    
    print("="*70 + "\n")
    
    # Per-class accuracy
    print("PER-CLASS ACCURACY:")
    print("-"*70)
    for label in class_names:
        total_class = sum(confusion[label].values())
        correct_class = confusion[label][label]
        acc = (correct_class / total_class * 100) if total_class > 0 else 0
        print(f"{label}: {correct_class}/{total_class} = {acc:.2f}%")
    print("="*70 + "\n")


def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python classify_audio.py <audio_file> [model_path]")
        print("  Test mode:   python classify_audio.py -test [model_path]")
        sys.exit(1)
    
    # Check for test flag
    if sys.argv[1] == '-test':
        model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
        model, class_names, label_mapping = load_model(model_path)
        
        if model is None:
            print("Failed to load model. Exiting.")
            sys.exit(1)
        
        test_accuracy(model, class_names)
    
    else:
        # Single file classification
        audio_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_PATH
        
        model, class_names, label_mapping = load_model(model_path)
        
        if model is None:
            print("Failed to load model. Exiting.")
            sys.exit(1)
        
        classify_audio(audio_path, model, class_names, verbose=True)


if __name__ == "__main__":
    main()
