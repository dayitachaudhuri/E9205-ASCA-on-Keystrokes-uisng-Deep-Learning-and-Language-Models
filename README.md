# 🛡️ Acoustic Side Channel Attack on Keystrokes using Deep Learning and Language Models

## 📚 Project Overview

This project focuses on enhancing keystroke prediction by leveraging deep learning models and language models to improve accuracy in **acoustic side channel attacks**. Acoustic side channel attacks exploit keystroke sounds to infer sensitive information such as passwords or typed text. Our approach combines a deep neural network (DNN) for classifying individual keystrokes and a language model to refine the predicted sequence of typed text, potentially increasing accuracy.

---

## 📄 Abstract

This project implements and adapts the methodology presented in the **IEEE Deep Learning-based Acoustic Side Channel Attack on Keyboards** paper, aimed at classifying keystrokes using audio recordings. Our primary objective is to enhance the accuracy of keystroke prediction by integrating a language model with a deep learning-based keystroke classification system.

The approach involves:

1. **Keystroke Classification:** Classifying individual keystrokes using a convolutional or recurrent neural network.
2. **Sequence Refinement:** Employing a language model (potentially transformer-based) to improve sequence prediction and correct classification errors.

We use datasets from **Kaggle** and the **Typing Behavior Dataset** to train and evaluate our models, addressing practical security concerns and exploring the limits of acoustic side channel attacks.

---

## 🎯 Motivation

The increasing vulnerability of keyboard inputs to **acoustic side channel attacks** poses a significant privacy risk. These attacks capture keystroke sounds using nearby microphones, enabling attackers to reconstruct typed text.

### Why This Matters:

- **High Threat Potential:** Attacks can be executed in public spaces like coffee shops or libraries, compromising sensitive information such as passwords.
- **Limitations of Traditional Security Measures:** Encryption and other traditional methods do not prevent passive audio leakage.
- **Advancing Security Research:** Understanding and improving attack models helps identify vulnerabilities and strengthen defenses.

### Key Inspirations:

- **"A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards"** demonstrated that a deep learning model (CoatNet) achieved 95% accuracy for smartphone-based keystroke recording and 93% via Zoom.
- **"PassGPT: Password Modeling and (Guided) Generation with Large Language Models"** showed that large language models (LLMs) can significantly improve password guessing, highlighting their potential in sequence prediction tasks.

---

## 🧩 Prior Work and Data Resources

### 📑 Key Papers:

1. **A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards (2023):**

   - Demonstrates the use of a deep learning model (CoatNet) to classify laptop keystrokes recorded by a smartphone.
   - Achieved 95% accuracy with smartphone recordings and 93% via Zoom.
2. **PassGPT: Password Modeling and (Guided) Generation with Large Language Models (2023):**

   - Explores the application of LLMs in password modeling and guided password generation.
   - Inspires our approach to integrating LLMs for sequence prediction.

---

### 📊 Datasets Used:

1. **Typing Behavior Dataset (Michigan State University, 2015):**

   - Audio and typing data from 45 participants collected in fixed and free-text typing sessions.
   - Includes acoustic signals, video data, and metadata for keystrokes.
2. **Kaggle “Keystrokes Noiseless Final” Dataset:**

   - Contains 593 `.wav` files representing keystroke sounds for all alphanumeric characters.
3. **Keystroke-Datasets (GitHub):**

   - Provides additional `.wav` files for specific keys, offering variety in acoustic patterns.

---

## 🛠️ Methodology

### 1. 📢 Data Collection and Preprocessing

- **Audio Cleaning:** Removing background noise and normalizing keystroke audio.
- **Feature Extraction:** Using Mel-Frequency Cepstral Coefficients (MFCCs) to convert audio signals into feature vectors.
- **Data Augmentation:** Applying time-shifting and noise injection for model robustness.

### 2. 🧠 Deep Learning Model for Keystroke Classification

- **Model Architecture:**
  - CNN/RNN-based model to classify keystroke audio data.
  - Input: MFCC features of keystroke audio.
  - Output: Classified key corresponding to each audio sample.
- **Optimization Techniques:**
  - Regularization and dropout to prevent overfitting.
  - Gradient descent with backpropagation for model training.

### 3. 🔡 Language Model for Sequence Refinement

- **Model Selection:**
  - Transformer-based language model or LSTM/GRU for sequence correction.
  - Autoregressive decoding for sequence prediction.
- **Error Correction:**
  - Refining the predicted keystroke sequence using language model outputs.
  - PassGPT-inspired techniques to predict probable key sequences in the context of the input text.

---

## 📈 Evaluation Metrics

- **Accuracy:** Percentage of correctly classified keystrokes.
- **Top-k Accuracy:** Percentage of true keys present in the top-k predictions.
- **Character Error Rate (CER):** Proportion of incorrect characters in predicted text.
- **Sequence Accuracy:** Correctness of the final predicted sequence.

---

## 👥 Team and Responsibilities

### Member 1

- 📚 **Literature Review:** Analyzing papers on acoustic side channel attacks and language modeling.
- 📊 **Data Collection and Preprocessing:** Organizing and cleaning audio datasets.
- 🤖 **Keystroke Classification Model:** Implementing and optimizing the deep learning model.

### Member 2

- 📚 **Literature Review:** Investigating LLMs for sequence prediction and password generation.
- 🎧 **Data Collection and Preprocessing:** Extracting MFCC features and preparing audio data.
- 🔡 **Language Model Integration:** Implementing a transformer-based or autoregressive language model.

### 🤝 **Collaborative Responsibilities**

- 📊 **Experimental Setup:** Designing and reviewing the experimental pipeline.
- ✍️ **Report Writing:** Co-authoring the final project report.
- 🗓️ **Weekly Progress Meetings:** Ensuring alignment on tasks and milestones.

---

## 📚 Course Relevance

This project applies several key concepts from the course:

- **Signal Processing Techniques:** Handling audio signals and extracting MFCC features.
- **Machine Learning Models:** Implementing and optimizing CNN/RNN models for classification.
- **Neural Network Optimization:** Using regularization, dropout, and backpropagation to improve model performance.
- **Language Models:** Applying transformer architectures and autoregressive decoding for sequence correction.

---

## 🚀 Getting Started

### 🔥 Installation

1. Clone the repository:

```bash
git clone https://github.com/dayitachaudhuri/E9_205
cd E9_205
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 📡 Data Preparation

- Download datasets from [MKA Dataset](https://data.mendeley.com/datasets/bpt2hvf8n3/4)
- Extract and organize audio files in the `data/` directory.

---

## 📂 Project Structure

```
├── data/                          # Raw and processed datasets
├── models/                        # Trained models
├── notebooks/                     # Jupyter notebooks for analysis
├── src/
│   ├── preprocess.py              # Audio preprocessing and feature extraction
│   ├── train_classifier.py        # Training script for keystroke classification
│   ├── language_model.py          # Language model integration
│   └── evaluate.py                # Evaluation and performance metrics
├── results/                       # Evaluation results and model outputs
└── README.md                      # Project documentation
```

---

## 🧪 Usage

### 🎧 Keystroke Classification

```bash
python src/train_classifier.py --data_dir data/ --epochs 50 --batch_size 32
```

### 🔡 Language Model Integration

```bash
python src/language_model.py --input_dir results/ --model_type transformer
```

### 📊 Evaluation

```bash
python src/evaluate.py --model_dir models/ --metrics accuracy cer
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

✅ **Happy Coding!** 🎉
