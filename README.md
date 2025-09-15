# Waste Classification using Transfer Learning (MobileNetV2)

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place images in `raw_dataset/` with subfolders for each class:
```
raw_dataset/
  ├── Recyclable/
  └── Non-Recyclable/
```

### 3. Split Dataset
```bash
python split_dataset.py
```

### 4. Train the Model
```bash
python train.py
```

### 5. Run Web App
```bash
streamlit run app.py
```
