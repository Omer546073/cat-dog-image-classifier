# Cats vs Dogs Classifier (Flask + MobileNetV2)

A beginner-friendly image classification web app: upload a photo, the model
predicts cat or dog.

## How it works

- A pretrained **MobileNetV2** (trained on ImageNet) is used as a frozen
  feature extractor — we only train a small classification "head" on top
  of it. This makes training fast even on a CPU.
- **Flask** serves an upload form and runs predictions on uploaded images.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
Download the Cats-vs-Dogs dataset from Kaggle and unzip it into a folder
named `data/` in this project (see chat for download instructions).

### 3. Organize the dataset
Kaggle datasets come in different folder layouts, so run this helper script
to sort everything into `data_organized/train/{cats,dogs}` and
`data_organized/val/{cats,dogs}`:
```bash
python organize_dataset.py
```
Check the printed counts to confirm it found a reasonable number of images
in each class. If it found 0 in either class, open the `data/` folder
yourself and check the structure, then adjust `find_images_by_class()` in
`organize_dataset.py` to match.

### 4. Train the model
```bash
python train_model.py
```
This will:
- Load images at 160x160 (smaller = faster on CPU)
- Train only the classification head (base MobileNetV2 stays frozen)
- Run for 8 epochs (~10-25 min on a typical CPU, depending on dataset size
  and your machine — feel free to lower EPOCHS in the script for a faster
  first run)
- Save the trained model to `model/cats_dogs_model.h5`

You should expect somewhere around 90-97% validation accuracy with this
setup, since MobileNetV2 already "knows" general image features.

### 5. Run the Flask app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser, upload a cat or dog photo,
and see the prediction.

## Tips for speeding up training further (CPU)
- Lower `EPOCHS` in `train_model.py` to 3-5 for a quick first test
- Use a subset of the dataset (e.g., 2,000 images per class) while you're
  iterating, then train on the full set once everything works
- Close other heavy applications while training

## Next steps / ideas to extend this project
- Add a confidence bar/chart in the UI
- Support drag-and-drop file upload
- Deploy it (Render, Railway, or PythonAnywhere all have free tiers)
- Swap in a different dataset (e.g., 10-class CIFAR-10) and make it
  multi-class instead of binary
- Add a "history" page showing past predictions using a small SQLite DB
