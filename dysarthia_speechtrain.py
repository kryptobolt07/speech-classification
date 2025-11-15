"""
Modified Speech Classification Training Script (Torgo Only)
Classifies audio into: Normal vs. Dysarthric (Torgo dataset)
Optimized for NVIDIA GPU

MODIFICATIONS INCLUDED:
1. Data Scope: Restricted to Normal (Control) and Dysarthric from Torgo dataset.
2. Stability Fixes (from original): 
    - matplotlib.use("Agg"): To prevent low-level graphical backend conflicts.
    - ResNet18Feature: Forcing weights=None to prevent sandbox issues during model loading/download.
    - DataLoader: Forcing num_workers=0 and pin_memory=False to prevent Linux shared memory/forking conflicts.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Force non-Qt backend to avoid graphical issues
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import torchvision.models as models # Import here for ResNet
from collections import Counter
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TORGO_PATH = os.path.join(BASE_PATH, "Torgo") # Only Torgo path is kept

RECORDING_DURATION_SEC = 3  # Duration for feature extraction
N_MELS = 128  # Mel spectrogram bins
FEATURE_SIZE = 128  # Target feature size (128x128)

# GPU Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA not available, using CPU")

print(f"Using device: {DEVICE}\n")

# --- DATA LOADING FUNCTIONS (Torgo Only) ---

def load_torgo_dataset(base_path):
    """Load audio files from TORGO dataset (Dysarthric and Control)"""
    audio_files = []
    labels = []
    
    if not os.path.exists(base_path):
        print(f"Warning: Torgo dataset not found at {base_path}")
        return audio_files, labels
    
    # F_Con (Female Control), M_Con (Male Control) -> Normal
    # F_Dys (Female Dysarthric), M_Dys (Male Dysarthric) -> Dysarthric
    subdirs = ['F_Con', 'F_Dys', 'M_Con', 'M_Dys']
    
    for subdir in tqdm(subdirs, desc="Loading Torgo Dataset"):
        current_path = os.path.join(base_path, subdir)
        if not os.path.exists(current_path):
            continue
            
        label = 'Dysarthric' if 'Dys' in subdir else 'Normal'
        
        for root, _, files in os.walk(current_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    audio_files.append(file_path)
                    labels.append(label)
    
    return audio_files, labels

def load_all_datasets():
    """Load Torgo dataset (Only Normal and Dysarthric)"""
    print("="*60)
    print("Loading Torgo Dataset (Normal vs. Dysarthric)")
    print("="*60)
    
    # Load Torgo dataset
    print("\n1. Loading Torgo Dataset...")
    torgo_files, torgo_labels = load_torgo_dataset(TORGO_PATH)
    print(f"   Loaded {len(torgo_files)} files from Torgo")
    
    all_audio_files = torgo_files
    all_labels = torgo_labels
    
    if not all_audio_files:
        raise ValueError("No audio files loaded from the Torgo path. Check configuration.")
    
    print(f"\nTotal files loaded: {len(all_audio_files)}")
    print(f"Label distribution:")
    label_counts = Counter(all_labels)
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    return all_audio_files, all_labels

# --- FEATURE EXTRACTION ---

class AudioFeatureExtractor(nn.Module):
    """Extract Mel-Spectrogram features from audio files"""
    def __init__(self, n_mels=128, duration=RECORDING_DURATION_SEC, sr=22050):
        super().__init__()
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
    
    def extract_mel_spectrogram(self, file_path):
        """Generate Normalized Mel-Spectrogram from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration, mono=True)
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=self.n_mels,
                n_fft=2048,
                hop_length=512
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-1 range
            if mel_spec_db.max() - mel_spec_db.min() > 0:
                mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            else:
                mel_spec_norm = mel_spec_db
            
            return mel_spec_norm
        except Exception as e:
            # print(f"Error processing {file_path}: {e}") # Keep commented out to avoid excessive output
            return None

def prepare_feature_vectors(audio_files, labels, batch_size=32):
    """
    Prepare 4D feature tensors (Mel-Spectrograms) for CNN training.
    Returns: 4D Feature Tensor (N, 1, 128, 128) and Labels Tensor
    """
    mel_extractor = AudioFeatureExtractor()
    mel_features = []
    dataset_labels = []
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    
    print(f"\nLabel mapping: {label_mapping}")
    print(f"Extracting features from {len(audio_files)} audio files...")
    
    failed_files = 0
    for file_path, label in tqdm(zip(audio_files, encoded_labels), 
                                  total=len(audio_files), 
                                  desc="Extracting Features"):
        mel_spec = mel_extractor.extract_mel_spectrogram(file_path)
        
        if mel_spec is not None:
            # Resize to target size (128x128)
            h, w = mel_spec.shape
            
            # Crop or pad to ensure 128x128
            if h > FEATURE_SIZE:
                mel_spec = mel_spec[:FEATURE_SIZE, :]
            if w > FEATURE_SIZE:
                mel_spec = mel_spec[:, :FEATURE_SIZE]
            
            # Pad if necessary
            pad_h = max(0, FEATURE_SIZE - mel_spec.shape[0])
            pad_w = max(0, FEATURE_SIZE - mel_spec.shape[1])
            
            if pad_h > 0 or pad_w > 0:
                mel_spec = np.pad(
                    mel_spec, 
                    ((0, pad_h), (0, pad_w)), 
                    mode='constant',
                    constant_values=0
                )
            
            # Convert to PyTorch Tensor, add channel dimension (1, 128, 128)
            mel_spec_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            
            mel_features.append(mel_spec_tensor)
            dataset_labels.append(label)
        else:
            failed_files += 1
    
    if failed_files > 0:
        print(f"Warning: {failed_files} files failed to process")
    
    # Stack into a single 4D Tensor (N, 1, 128, 128)
    if len(mel_features) == 0:
        raise ValueError("No valid features extracted from audio files!")
    
    X = torch.stack(mel_features)
    y = torch.tensor(dataset_labels, dtype=torch.long)
    
    return X, y, label_mapping

# --- MODEL DEFINITION ---

class ResNet18Feature(nn.Module):
    """
    Modified ResNet18 for Mel-Spectrogram classification (1-channel input).
    Dynamically set num_classes (will be 2 for Normal/Dysarthric).
    """
    def __init__(self, num_classes=2): # Default to 2 classes
        super().__init__()
        # Load pre-trained ResNet18
        try:
            # --- MODIFICATION: Force weights=None to avoid sandbox/download issue ---
            self.resnet = models.resnet18(weights=None)
            print("Initialized ResNet18 without pre-trained weights (to prevent sandbox error)")
            # -----------------------------------------------------------------------
            
        except Exception as e:
            # Fallback will also use no weights
            print(f"Could not load torchvision (install failure?): {e}")
            self.resnet = models.resnet18(weights=None)
            print("Initialized ResNet18 without pre-trained weights")
        
        # Modify the first layer for single-channel (grayscale) input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify last layer for classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# --- TRAINING FUNCTIONS ---

def train_resnet_model(X, y, label_mapping, test_size=0.2, random_state=42, 
                      epochs=50, batch_size=64, learning_rate=0.001):
    """
    Train ResNet18 model using PyTorch with GPU acceleration.
    """
    num_classes = len(label_mapping)
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {num_classes}")
    print(f"Test size: {test_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Move data to GPU if available (temporarily for splitting)
    if torch.cuda.is_available():
        X = X.to(DEVICE)
        y = y.to(DEVICE)
    
    # Split data (use CPU numpy for robust splitting)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(), 
        test_size=test_size, 
        random_state=random_state,
        stratify=y.cpu().numpy()
    )
    
    # Move split data back to device
    X_train = torch.from_numpy(X_train_np).float().to(DEVICE)
    X_test = torch.from_numpy(X_test_np).float().to(DEVICE)
    y_train = torch.from_numpy(y_train_np).long().to(DEVICE)
    y_test = torch.from_numpy(y_test_np).long().to(DEVICE)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Setup DataLoaders with stability modifications
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # --- MODIFICATION: Set num_workers=0 and pin_memory=False explicitly ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0, 
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, 
        pin_memory=False
    )
    # ------------------------------------------------------------------------
    
    # Model, Loss, Optimizer
    model = ResNet18Feature(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training Loop
    print(f"Training ResNet18 on {DEVICE} for {epochs} epochs...\n")
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss = val_loss / len(test_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> Saved best model (Val Loss: {val_loss:.4f})")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("\nClassification Report:")
    print(class_report)
    
    # Return the model on CPU for general use
    model.to('cpu')
    
    return model, class_report, conf_matrix, class_names, train_losses, val_losses

def plot_results(conf_matrix, class_names, train_losses, val_losses):
    """Plot confusion matrix and training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion Matrix
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cmap='Blues',
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix - ResNet18 CNN (Normal vs. Dysarthric)')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Training Curves
    axes[1].plot(train_losses, label='Train Loss')
    axes[1].plot(val_losses, label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results_torgo.png', dpi=300, bbox_inches='tight')
    print("\nResults saved to 'training_results_torgo.png'")
    # Note: plt.show() is commented out to prevent another potential GUI conflict

# --- INFERENCE FUNCTIONS ---

def classify_new_audio(audio_path, model, label_mapping):
    """
    Classify a new audio file using the trained model.
    """
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    feature_extractor = AudioFeatureExtractor(n_mels=N_MELS, duration=RECORDING_DURATION_SEC)
    
    print(f"Processing audio file: {audio_path}")
    
    # Feature extraction
    mel_spec = feature_extractor.extract_mel_spectrogram(audio_path)
    
    if mel_spec is None:
        return "Classification Failed: Feature extraction error or unsupported file."
    
    # Resize and pad (must match training logic)
    h, w = mel_spec.shape
    if h > FEATURE_SIZE:
        mel_spec = mel_spec[:FEATURE_SIZE, :]
    if w > FEATURE_SIZE:
        mel_spec = mel_spec[:, :FEATURE_SIZE]
    
    pad_h = max(0, FEATURE_SIZE - mel_spec.shape[0])
    pad_w = max(0, FEATURE_SIZE - mel_spec.shape[1])
    
    if pad_h > 0 or pad_w > 0:
        mel_spec = np.pad(
            mel_spec, 
            ((0, pad_h), (0, pad_w)), 
            mode='constant',
            constant_values=0
        )
    
    # Preprocessing for CNN
    X_new = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
    X_new = X_new.to(DEVICE)
    
    # Prediction
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        outputs = model(X_new)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities.data, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score

# --- MAIN EXECUTION ---

def main():
    print("="*60)
    print("Speech Classification Training (Normal vs. Dysarthric)")
    print("Using: Torgo Dataset Only")
    print("="*60)
    
    # Step 1: Load Torgo dataset only
    audio_files, labels = load_all_datasets()
    
    if len(audio_files) == 0:
        print("ERROR: No audio files found. Please check Torgo dataset path.")
        return
    
    # Step 2: Prepare feature vectors
    print(f"\n{'='*60}")
    print("Feature Extraction")
    print(f"{'='*60}")
    X, y, label_mapping = prepare_feature_vectors(audio_files, labels)
    print(f"Feature tensor shape: {X.shape}")
    print(f"Labels tensor shape: {y.shape}")
    
    # Step 3: Train model
    print(f"\n{'='*60}")
    print("Model Training")
    print(f"{'='*60}")
    model, report, conf_matrix, class_names, train_losses, val_losses = train_resnet_model(
        X, y, label_mapping, 
        epochs=50, 
        batch_size=64,
        learning_rate=0.001
    )
    
    # Step 4: Plot results
    print(f"\n{'='*60}")
    print("Generating Plots")
    print(f"{'='*60}")
    plot_results(conf_matrix, class_names, train_losses, val_losses)
    
    # Step 5: Save model
    final_model_path = 'speech_classifier_torgo_only.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_mapping': label_mapping,
        'class_names': class_names
    }, final_model_path)
    print(f"\nModel saved to '{final_model_path}'")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Starting training script...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check for required PyTorch/torchvision
    try:
        if 'torchvision' not in sys.modules:
            import torchvision # Attempt to import to check
    except ImportError:
        print("\nFATAL ERROR: PyTorch and/or Torchvision is required but not installed.")
        print("Please install with: pip install torch torchvision torchaudio")
        sys.exit(1)
        
    try:
        main()
    except Exception as e:
        print(f"\nCRITICAL ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)