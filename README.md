# 📦 Scene Text Detection & Recognition (YOLO + CRNN)

An end-to-end Optical Character Recognition (OCR) framework designed to detect and transcribe text from natural scene images. 

### Key Features:
- **Text Detection:** Powered by YOLO to locate bounding boxes around scene text.
- **Text Recognition:** Custom CRNN architecture utilizing a ResNet backbone, bi-directional GRU layers, and CTC decoding.
- **Smart Reading Order:** Automatically sorts detected text bounding boxes top-to-bottom, left-to-right to maintain natural reading flow.
- **Interactive UI:** Built-in Gradio web application for seamless image upload, visualization, and text extraction.

![alt text](https://github.com/quangvu32/scene_text_recognition/blob/main/asset/1.jpg?raw=true)

## 📥 Pretrained Weights

Please download the trained model weights before running the application:

| Model | Size | Download Link |
| :--- | :--- | :--- |
| **YOLO Text Detector** (`best.pt`) | ~14 MB | [Download](https://drive.google.com/drive/folders/1wAa-aSI1fgBSPKQO8gwlfpMDte1GvwO1?usp=drive_link) |
| **CRNN Text Recognizer** (`ocr_crnn.pt`) | ~200 MB | [Download](https://drive.google.com/drive/folders/1DXG3fmV4FaheaRYhyxDQQoXY50vAIx9y?usp=drive_link) |

Place the downloaded `.pt` files into the root directory of the project.
