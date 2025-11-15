import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# --- NEW IMPORTS FOR AUDIO RECORDING/PROCESSING ---
AUDIO_RECORDING_ENABLED = False
try:
    import sounddevice as sd
    from scipy.io.wavfile import write
    AUDIO_RECORDING_ENABLED = True
except ImportError:
    print("Warning: 'sounddevice' or 'scipy' not fully available. Audio recording feature will be disabled.")
# ------------------------------------------------

# Define device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- CONFIGURATION (ORIGINAL PATH RETAINED) ---
BASE_PATH = "Torgo"  
RECORDING_DURATION_SEC = 3

# --- CORE FUNCTIONS ---

def load_audio_files(base_path):
    """
    Load audio files from TORGO dataset directories
    """
    audio_files = []
    labels = []
    
    subdirs = ['F_Con', 'F_Dys', 'M_Con', 'M_Dys']
    
    # Use os.walk for robust traversal
    for subdir in tqdm(subdirs, desc="Processing Directories"):
        current_path = os.path.join(base_path, subdir)
        label = 'Dysarthric' if 'Dys' in subdir else 'Control'
        
        # Check if directory exists before walking
        if not os.path.isdir(current_path):
            print(f"Warning: Directory not found: {current_path}. Skipping.")
            continue
            
        for root, dirs, files in os.walk(current_path):
            for file in tqdm(files, desc=f"Scanning {subdir}", leave=False):
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    audio_files.append(file_path)
                    labels.append(label)
    
    return audio_files, labels

class AudioFeatureExtractor(nn.Module):
    # ... (Extractor code remains the same) ...
    def __init__(self, n_mels=128, duration=RECORDING_DURATION_SEC):
        super().__init__()
        self.n_mels = n_mels
        self.duration = duration
    
    def extract_mel_spectrogram(self, file_path):
        """
        Generate Normalized Mel-Spectrogram from audio file
        """
        try:
            y, sr = librosa.load(file_path, duration=self.duration)
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-1 range
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            
            return mel_spec_norm
        except Exception as e:
            # print(f"Error processing {file_path}: {e}")
            return None

class ResNet18Feature(nn.Module):
    # ... (ResNet18Feature code remains the same) ...
    def __init__(self, num_classes=2):
        super().__init__()
        try:
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        except:
             import torchvision.models as models
             print("Initializing ResNet18 without pre-trained weights (online loading failed).")
             self.resnet = models.resnet18(weights=None)
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def prepare_feature_vectors(audio_files, labels):
    """
    Prepare 4D feature tensors (Mel-Spectrograms) for CNN training.
    """
    mel_extractor = AudioFeatureExtractor()
    mel_features = []
    dataset_labels = []
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    for file_path, label in tqdm(zip(audio_files, encoded_labels), 
                                  total=len(audio_files), 
                                  desc="Extracting Features"):
        mel_spec = mel_extractor.extract_mel_spectrogram(file_path)
        
        if mel_spec is not None:
            # Ensure consistent feature size (128x128)
            mel_spec_resized = mel_spec[:128, :128]
            mel_spec_resized = np.pad(
                mel_spec_resized, 
                ((0, 128 - mel_spec_resized.shape[0]), (0, 128 - mel_spec_resized.shape[1])), 
                mode='constant'
            )
            
            mel_spec_tensor = torch.from_numpy(mel_spec_resized).float().unsqueeze(0)
            
            mel_features.append(mel_spec_tensor)
            dataset_labels.append(label)

    # --- ADDED CHECK TO PREVENT CRASH ---
    if not mel_features:
        raise RuntimeError(
            "Feature extraction failed for ALL audio files. "
            "This likely means the BASE_PATH is incorrect, or no valid .wav files were found."
        )
    # ------------------------------------
    
    # Stack into a single 4D Tensor (N, 1, 128, 128)
    X = torch.stack(mel_features)
    y = torch.tensor(dataset_labels, dtype=torch.long)
    
    return X, y

def train_resnet_model(X, y, test_size=0.2, random_state=42, epochs=10, batch_size=32):
    # ... (train_resnet_model code remains the same) ...
    # 1. Split data 
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(), 
        test_size=test_size, 
        random_state=random_state,
        stratify=y.cpu().numpy()
    )
    X_train = torch.from_numpy(X_train_np).float()
    X_test = torch.from_numpy(X_test_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    y_test = torch.from_numpy(y_test_np).long()
    
    # 2. Setup DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model, Loss, Optimizer
    model = ResNet18Feature(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    print(f"Training ResNet18 on {DEVICE} for {epochs} epochs...")
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 5. Evaluation
    model.eval()
    all_preds = []
    y_test_gpu = y_test.to(DEVICE)
    X_test_gpu = X_test.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(X_test_gpu)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        
    # Generate reports
    y_pred = np.array(all_preds)
    y_true = y_test_gpu.cpu().numpy()
    
    class_names = ['Control', 'Dysarthric']
    class_report = classification_report(y_true, y_pred, target_names=class_names)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    model.to('cpu')
    return model, class_report, conf_matrix

def plot_confusion_matrix(conf_matrix, class_names):
    # ... (Plotting code remains the same) ...
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        xticklabels=class_names, 
        yticklabels=class_names,
        cmap='Blues'
    )
    plt.title('Confusion Matrix - ResNet18 CNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# --- REAL-TIME CLASSIFICATION FUNCTIONS ---
# ... (record_audio and classify_new_audio remain the same) ...

def record_audio(filename='user_audio.wav', duration=RECORDING_DURATION_SEC, sample_rate=16000):
    if not AUDIO_RECORDING_ENABLED: return None
    print(f"\nStarting recording for {duration} seconds... Please speak clearly into your microphone.")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        write(filename, sample_rate, recording)
        print(f"Recording finished and saved to {filename}")
        return filename
    except Exception as e:
        print(f"An error occurred during recording. Make sure your microphone is enabled/selected: {e}")
        return None

def classify_new_audio(audio_path, model):
    class_names = ['Control', 'Dysarthric']
    feature_extractor = AudioFeatureExtractor(n_mels=128, duration=RECORDING_DURATION_SEC) 
    print(f"Processing audio file: {audio_path}")
    mel_spec = feature_extractor.extract_mel_spectrogram(audio_path)
    if mel_spec is None: return "Classification Failed: Feature extraction error or unsupported file."
    mel_spec_resized = mel_spec[:128, :128]
    mel_spec_resized = np.pad(
        mel_spec_resized, 
        ((0, 128 - mel_spec_resized.shape[0]), (0, 128 - mel_spec_resized.shape[1])), 
        mode='constant'
    )
    X_new = torch.from_numpy(mel_spec_resized).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(X_new)
        _, predicted_idx = torch.max(outputs.data, 1)
    model.to('cpu')
    return class_names[predicted_idx.item()]

# --- MAIN EXECUTION ---

def main(base_path):
    # Step 1: Load Audio Files
    print("Step 1: Loading Audio Files")
    audio_files, labels = load_audio_files(base_path)
    
    # --- DIAGNOSTIC PRINT ---
    print(f"Total audio files found: {len(audio_files)}\n")
    # If the number above is 0, the BASE_PATH is the problem.
    if len(audio_files) == 0:
        print("CRITICAL ERROR: No audio files were found. Please check that the BASE_PATH is correct.")
        print(f"CURRENT PATH: {base_path}")
        print("Exiting...")
        return
    # ------------------------

    # Step 2: Prepare Feature Vectors (for CNN)
    print("Step 2: Preparing Feature Tensors")
    # This call will now raise the more descriptive RuntimeError if it still fails
    X, y = prepare_feature_vectors(audio_files, labels)
    print(f"Feature tensor shape: {X.shape}")
    print(f"Labels tensor shape: {y.shape}\n")
    
    # Step 3: Train ResNet Model
    print("Step 3: Training ResNet18 Model")
    model, report, conf_matrix = train_resnet_model(X, y, epochs=10)
    
    # Step 4: Print Results
    print("\nClassification Report:")
    print(report)
    
    # Step 5: Plot Confusion Matrix
    plot_confusion_matrix(conf_matrix, ['Control', 'Dysarthric'])
    
    # Step 6: Record and Classify New Audio
    print("\n" + "="*50)
    print("Step 6: Real-Time Audio Recording and Classification")
    
    if not AUDIO_RECORDING_ENABLED:
        print("Audio recording feature is disabled. Skipping step.")
        print("To enable, install 'sounddevice' and 'scipy'.")
        print("="*50)
        return
    
    new_audio_filename = "recorded_speech_for_classification.wav"
    recorded_path = record_audio(filename=new_audio_filename, duration=RECORDING_DURATION_SEC)
    
    if recorded_path:
        prediction = classify_new_audio(recorded_path, model)
        print(f"\nClassification Result: The model predicts the audio is **{prediction}**")
        try:
            os.remove(recorded_path)
        except Exception as e:
            print(f"Could not remove temporary file: {e}")
            
    print("="*50)


if __name__ == "__main__":
    main(BASE_PATH)