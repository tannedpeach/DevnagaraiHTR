# Devanagari Handwritten Text Recognition (HTR)

A deep learning-based handwritten text recognition system specifically designed for Devanagari script (Hindi, Marathi, Sanskrit, and other Indian languages). The system uses a CNN-RNN-CTC neural network architecture to accurately recognize handwritten text from images.

## Features

- **CNN-RNN-CTC Architecture**: Combines Convolutional Neural Networks for feature extraction, Recurrent Neural Networks for sequence modeling, and Connectionist Temporal Classification for sequence-to-sequence learning
- **Multiple Decoding Strategies**: 
  - Best Path Decoding (fastest)
  - Beam Search Decoding (more accurate)
  - Word Beam Search Decoding (language model-based)
- **Web Interface**: Flask-based web application for easy image upload and text recognition
- **Training & Validation**: Complete pipeline for training on custom datasets with automatic train/validation split (90/10)
- **Real-time Prediction**: Get confidence scores along with recognized text

## Architecture

The model consists of three main components:

1. **CNN Layers**: 5 convolutional layers for feature extraction from input images
   - Input: 128x32 grayscale images
   - Kernel sizes: [5, 5, 3, 3, 3]
   - Feature maps: [1, 32, 64, 128, 128, 256]

2. **RNN Layers**: Bidirectional LSTM layers for sequence modeling
   - 2 stacked LSTM layers
   - 256 hidden units per layer
   - Bidirectional processing for better context understanding

3. **CTC Layer**: Connectionist Temporal Classification for sequence-to-sequence alignment
   - Handles variable-length sequences
   - No explicit segmentation required

## Requirements

```bash
# Core dependencies
tensorflow==1.x  # (TensorFlow 1.x required for this implementation)
opencv-python
numpy
editdistance
flask

# For training
codecs
```

## Getting Started

### 1. Training a Model

```bash
# Train the model
python main.py --train

# Validate the model
python main.py --validate

# Train with beam search decoder
python main.py --train --beamsearch
```

### 2. Inference on a Single Image

```bash
# Run inference with default decoder (BestPath)
python main.py

# Run inference with beam search
python main.py --beamsearch

# Run inference with word beam search
python main.py --wordbeamsearch
```

### 3. Web Interface

```bash
# Start the Flask web server
python upload.py

# Access the application at http://localhost:4555
```

Upload an image of handwritten Devanagari text through the web interface and select your preferred decoding method.

## Project Structure

```
DevnagaraiHTR/
│
├── main.py                 # Main training and inference script
├── Model.py                # Neural network architecture (CNN-RNN-CTC)
├── DataLoader.py           # Dataset loading and batch generation
├── SamplePreprocessor.py   # Image preprocessing utilities
├── upload.py               # Flask web application
│
├── data/                   # Training data directory
│   └── full.txt           # Ground truth labels file
│
└── model/                  # Saved model directory
    ├── charList.txt       # Character list (vocabulary)
    └── accuracy.txt       # Validation accuracy log
```

## Data Format

The training data should be organized with:
- Images in the `data/` directory
- A `full.txt` file containing space-separated filename and ground truth text pairs:

```
image1.jpg नमस्ते
image2.jpg हिंदी
image3.jpg भारत
```

## Model Performance

The model tracks:
- **Character Error Rate (CER)**: Measures accuracy at the character level
- **Word Accuracy**: Percentage of correctly recognized words
- **Confidence Scores**: Probability estimates for each prediction

Training includes early stopping mechanism that halts after 5 epochs without improvement.

## Configuration

Key parameters in `Model.py`:
- `batchSize = 25`: Number of samples per batch
- `imgSize = (128, 32)`: Input image dimensions
- `maxTextLen = 32`: Maximum text length supported
- `learningRate = 0.0001`: Training learning rate (RMSProp optimizer)

## Web Interface Features

- Upload images in JPG or PNG format
- Choose decoding strategy (BestPath or BeamSearch)
- View recognized text with confidence scores
- Timestamp-based image storage for tracking

## Contributing

Contributions are welcome! Areas for improvement:
- Upgrading to TensorFlow 2.x
- Adding data augmentation techniques
- Implementing attention mechanisms
- Support for multiple languages
- Improving preprocessing pipeline

## License

This project is open-source and available for educational and research purposes.

## Acknowledgments

- Built using TensorFlow's CTC implementation
- Inspired by handwriting recognition research in deep learning
- Designed for Indian language script recognition

## Contact

For questions or collaboration opportunities, please reach out through GitHub issues.

---

**Note**: This project was developed as an AI/ML learning project demonstrating end-to-end implementation of a handwritten text recognition system.
