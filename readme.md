# Mosaic Image Engine & Evaluation

A Python-based application that transforms standard photographs into detailed artistic photo mosaics using a custom-generated tile library, vectorized NumPy operations, K-Means color quantization, and perceptual LAB color-space matching. The project also features an interactive **Gradio** web interface and custom-built performance evaluation metrics (MSE and SSIM).

---

## 🚀 Project Structure

```text
├── main.py              # Gradio web interface & main pipeline wrapper
├── preprocessing.py     # Image cropping, area resizing, and K-Means quantization
├── mosaic_engine.py     # Grid division, tensor block averaging, and LAB tile mapping
├── performance.py       # Pure NumPy implementation of MSE and SSIM metrics
├── config.py            # Centralized project hyperparameters
└── tiles/               # Directory containing custom source image tiles