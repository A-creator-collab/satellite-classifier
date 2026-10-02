
# satellite-classifier

Fine-tunes a ResNet-18 to classify 64x64 Sentinel-2 RGB patches from EuroSAT into 10 land-use classes. Includes training, evaluation, GradCAM overlays, a FastAPI endpoint, and a Streamlit app.

## Results

ResNet-18, two-phase fine-tuning, 15 epochs total (3 head, 12 fine-tune).

- Best validation accuracy: 98.54%
- Dataset: EuroSAT RGB, 27,000 images, 10 classes, 64x64 pixels
- Split: 70% train / 15% validation / 15% test, stratified
- Training time: ~70 minutes on Windows 11, CPU-only

Run the commands below to reproduce these numbers.

<img width="1920" height="955" alt="stremlit-demo" src="https://github.com/user-attachments/assets/56df9275-666f-4fd2-95f2-d1acfe706179" />

## Dataset

EuroSAT RGB: 10 land-use classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake), 27,000 JPEG images at 64x64 pixels.

The original host `madm.dfki.de` is no longer reliably reachable. Download the dataset manually from one of these mirrors and place the extracted class folders under `data/raw/EuroSAT_RGB/`:

- Hugging Face: https://huggingface.co/datasets/torchgeo/eurosat
- Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

Required layout:

```
data/raw/EuroSAT_RGB/
    AnnualCrop/
    Forest/
    HerbaceousVegetation/
    Highway/
    Industrial/
    Pasture/
    PermanentCrop/
    Residential/
    River/
    SeaLake/
```

Each folder must contain `.jpg` files. Do not use the multispectral `.tif` variant (13 bands); the model and the data loader expect 3-channel RGB.

The training script checks for this layout first and skips the download if the folder is present.

## Install

Run from the repository root. Tested on Windows 11 with Python 3.13. CI runs on Python 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config config.yaml
```

Reads `config.yaml`, loads the dataset from `data/raw/EuroSAT_RGB/`, creates a stratified train/validation/test split saved to `outputs/splits.json`, trains the classification head with a frozen backbone for 3 epochs, then fine-tunes the full model for 12 epochs at a lower learning rate. Saves the best checkpoint to `outputs/best_model.pt`.

## Evaluate

```bash
python -m src.evaluate --config config.yaml
```

Writes `outputs/metrics.json` and `outputs/confusion_matrix.png`. Metrics: overall accuracy, Cohen's kappa, macro F1, per-class F1.

## Predict

```bash
python -m src.predict data/raw/EuroSAT_RGB/Forest/Forest_1.jpg --config config.yaml
```

Replace the path with any image from the dataset. Prints the predicted class and probabilities over all 10 classes.

## GradCAM

```bash
python -m src.gradcam --config config.yaml
```

Writes overlays to `outputs/gradcam/`. Use these to check that the model attends to land features rather than edges or noise.

## API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Test with a real image path:

```bash
# Windows PowerShell
curl.exe -X POST -F "file=@data/raw/EuroSAT_RGB/Forest/Forest_1.jpg" http://localhost:8000/predict

# Linux/macOS
curl -X POST -F "file=@data/raw/EuroSAT_RGB/Forest/Forest_1.jpg" http://localhost:8000/predict
```

On PowerShell, use `curl.exe` (not `curl`) to bypass the `Invoke-WebRequest` alias.

Interactive docs: http://localhost:8000/docs

## Streamlit

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501. Upload a 3-channel `.jpg` or `.png`. Uploading a 13-band `.tif` will raise `PIL.UnidentifiedImageError` because the model expects RGB.

## Docker (optional)

Requires Docker Desktop. Not needed for the API or Streamlit demos above.

```bash
docker build -t satellite-classifier .
# Windows PowerShell
docker run --rm -p 8000:8000 -v "${PWD}/outputs:/app/outputs" satellite-classifier
# Linux/macOS
docker run --rm -p 8000:8000 -v "$(pwd)/outputs:/app/outputs" satellite-classifier
```

The image expects `outputs/best_model.pt` to exist on the host before the volume mount. Train first.

## Project structure

```
satellite-classifier/
├── config.yaml
├── requirements.txt
├── src/
│   ├── data.py       dataset, splits, transforms
│   ├── model.py      ResNet-18 with a 10-way head
│   ├── train.py      two-phase fine-tuning loop
│   ├── evaluate.py   accuracy, kappa, macro F1, confusion matrix
│   ├── predict.py    single-image CLI prediction
│   └── gradcam.py    GradCAM overlays
├── api/              FastAPI app
├── app/              Streamlit app
├── tests/            pytest tests
├── outputs/          checkpoints, splits, metrics, GradCAM images
└── .github/workflows/ci.yml
```

## Configuration

All paths, hyperparameters, and seeds live in `config.yaml`.

Key fields:

- `data.url` — used only if the dataset is not already present. Set to a working mirror or `local` if you placed the folder manually.
- `data.input_size` — 64, matching EuroSAT's native resolution.
- `data.num_workers` — 0 works best on Windows for this dataset size.
- `train.epochs_head` / `train.epochs_finetune` — 3 and 12.
- `train.lr_head` / `train.finetune_lr_factor` — head learning rate and the multiplier applied to the backbone during fine-tuning.

## Reproducibility notes

- Seeds for Python, NumPy, and PyTorch come from `config.yaml`.
- cuDNN is set to deterministic mode and `torch.use_deterministic_algorithms` is enabled with warnings only, so some CUDA operations may still vary between runs.
- The split is stratified and stored in `outputs/splits.json`.
- Class weights are computed from the training split only.
- On CPU-only Windows, training takes roughly 70 minutes end to end.

## License

MIT. See `LICENSE`.
```
