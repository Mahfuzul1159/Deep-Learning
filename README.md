# 🧠 Deep Learning

This repository contains deep learning projects focused on real-world applications such as object detection, image classification, and model integration. The featured project demonstrates a complete pipeline for **Bangladeshi currency detection and recognition** using **YOLOv8** for detection and **ResNet50** for classification.

---

## 💼 Project: Bangladeshi Currency Recognition (YOLOv8 + ResNet50)

### 📝 Overview

This project presents a hybrid deep learning system that:

- **Detects** Bangladeshi currency notes from an input image using YOLOv8.
- **Crops** the detected currency region.
- **Classifies** the denomination using a fine-tuned ResNet50 model.
- **Visualizes** the prediction results, accuracy/loss curves, confusion matrix, and classification report.

It is designed for **real-time** currency recognition and optimized for performance and accuracy.

---

## 🚀 Features

✅ Object Detection using YOLOv8  
✅ Image Classification using ResNet50  
✅ Custom dataset for Bangladeshi currency (2–1000 Taka)  
✅ Real-time prediction on user-uploaded images  
✅ Data augmentation & visualization  
✅ Evaluation: Confusion matrix, ROC curve, PR curve  
✅ Integrated end-to-end pipeline in a single notebook

---

## 🧰 Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Ultralytics YOLOv8
- Matplotlib
- scikit-learn
- Google Colab / Jupyter Notebook

---

## 📁 Repository Structure

```
Deep-Learning/
├── YOLOandRESNET_final.ipynb       # Full pipeline notebook
├── README.md                       # Project documentation
├── dataset/
│   ├── Detection/                  # YOLOv8 dataset structure
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── Recognition/               # ResNet dataset structure
│       ├── train/
│       ├── val/
│       └── test/
├── models/                         # (Optional) Trained weights and saved models
└── sample_images/                 # Example input/output predictions
```

---

## 🛠️ How to Use

### ✅ Clone the Repository
```bash
git clone https://github.com/your-username/Deep-Learning.git
cd Deep-Learning
```

### ✅ Open the Notebook
You can run the notebook in **Google Colab** or **Jupyter Notebook**.

📌 Make sure to upload:
- Your trained YOLOv8 weights (`best.pt`)
- Your trained ResNet model (`resnet_taka.h5`)
- A test image to predict

---

## 📊 Evaluation Metrics

- 📈 Training vs. Validation Accuracy & Loss
- 📊 Confusion Matrix
- 📃 Classification Report (Precision, Recall, F1-score)
- 🧪 ROC & Precision-Recall Curve

---

## 📷 Example Output

> *(Upload an example image here and show bounding box + predicted denomination)*

---

## 🔮 Future Work

- Expand dataset with more currency types and real-world conditions
- Integrate OCR to recognize serial numbers on notes
- Optimize model using TensorRT for faster real-time inference
- Deploy as a mobile/web app for public use
- Add multi-language support for broader usability

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- Custom dataset of Bangladeshi currency

---

## 👨‍💻 Author

**Mahfuzul Islam**  
🎓 Student, North Western University, Khulna  
🌐 GitHub: [@your-username](https://github.com/your-username)  
📧 Email: your-email@example.com

---

## ⭐ If you found this helpful, give it a star and share!
