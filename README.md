# fish-species-classification
A Fish Species Classification project using a fine-tuned Vision Transformer (ViT) model to classify fish images into 15 different species. Built with Python, Hugging Face Transformers, and Streamlit, using deep learning and image classification techniques.
🐟 Fish Species Classification Using Vision Transformer (ViT)

Project Overview

Fish Species Classification is a deep learning project that uses a fine-tuned Vision Transformer (ViT) model to identify fish species from images.

The application allows users to upload a fish image or capture one using their camera. The trained model analyzes the image and predicts which of the 15 fish species it most closely matches.

The project combines Deep Learning, Computer Vision, Hugging Face Transformers, and Streamlit.

Problem Statement

Identifying fish species manually can be difficult, especially when different species have similar physical appearances.

The purpose of this project is to develop an image classification application that uses deep learning to classify fish images into 15 different categories.

Project Objectives

* Prepare and explore a fish image dataset.
* Preprocess images for deep learning.
* Use a pretrained Vision Transformer model.
* Fine-tune the model for 15 fish species.
* Evaluate the trained model using validation and test datasets.
* Host the trained model on Hugging Face.
* Develop an interactive Streamlit application.
* Deploy the application online.

Dataset

The project uses a fish image dataset provided in the archive Fish set 4.zip.

* Total images: 561
* Number of classes: 15
* Training images: 386
* Validation images: 75
* Testing images: 100

Fish Species Classes

1. Jinnaok : Top shell kator
2. Alwahor
3. Black pamplet
4. Chanchu
5. Gorr
6. Jinga : madag
7. Koh prosh
8. Konad
9. Lujar
10. Mah pari : Safi
11. Mushko
12. Sawaso
13. Shurr
14. Soli
15. White pamplet

Model Information

The project uses the pretrained Vision Transformer (ViT) model.

* Base model: google/vit-base-patch16-224
* Framework: Hugging Face Transformers
* Architecture: Vision Transformer
* Input image size: 224 × 224
* Number of output classes: 15

The original classification head was adapted for the 15 fish species, and the model was fine-tuned using the prepared dataset.

Training Configuration

Parameter	Value
Epochs	10
Batch size	8
Learning rate	0.00002
Image size	224 × 224
Model	ViT-Base-Patch16-224
Number of classes	15

Image Preprocessing

The images were prepared using the following steps:

* Resizing and cropping images to 224 × 224 pixels.
* Normalizing images using the pretrained image processor.
* Applying data augmentation to training images.
* Using separate training, validation, and testing datasets.

Training augmentation included random resized cropping, horizontal flipping, and brightness/contrast adjustment.

Model Performance

The trained model achieved the following results:

Evaluation	Accuracy
Best validation accuracy	82.67%
Test accuracy	82%

The test dataset contained 100 images, and the model correctly classified approximately 82 of them.

These results are from the completed training and evaluation experiments.

Hugging Face Model

The fine-tuned model is hosted on Hugging Face.

Model repository:

https://huggingface.co/BismaLJ/fish_species_classification

The Streamlit application downloads the model and image processor directly from this public repository.

Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Hub
* Vision Transformer (ViT)
* Streamlit
* Pillow
* Google Colab

Application Features

* Upload an image from your device.
* Capture an image using your camera.
* Classify images into 15 fish species.
* Display the predicted species.
* Display prediction confidence.
* Show the top five predicted classes.

Project Structure

fish-species-classification/
│
├── app.py
├── requirements.txt
└── README.md

Installation and Usage

1. Clone the repository

git clone https://github.com/YOUR-GITHUB-USERNAME/fish-species-classification.git

2. Open the project directory

cd fish-species-classification

3. Install dependencies

pip install -r requirements.txt

4. Run the Streamlit application

streamlit run app.py

The application will open in your browser.

Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Upload the project files to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this project repository.
5. Select app.py as the main file.
6. Deploy the application.

Important Note

This is a 15-class image classification model. It predicts the closest matching fish category from the classes it was trained on. It is not guaranteed to correctly identify images outside these 15 fish categories.

Author

Bisma Lal Jan

BS Information Technology
University of Gwadar

Project Type

University Deep Learning Project
