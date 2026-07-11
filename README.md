# Fingerprint Gender Classifier AI

This project trains and tests a deep learning AI model (ResNet18) to classify the gender of a person based purely on their fingerprint ridge patterns. 

## Features
* **Custom Dataset Loading:** Integrates directly with the Kaggle SOCOFing dataset to parse real fingerprint images and extract binary gender labels.
* **Transfer Learning:** Uses a modified PyTorch ResNet18 model fine-tuned specifically on fingerprint imagery.
* **Optical Photo Preprocessing:** Includes an OpenCV pipeline (`test_my_thumb.py`) to extract fingerprint ridges from standard smartphone optical photos, closing the domain gap between regular photos and biometric scans.

## Files
* `train_fingerprints.py`: Downloads the SOCOFing dataset, builds the AI architecture, and trains the model.
* `test_fingerprint.py` / `quick_test.py`: Evaluates the model on raw SOCOFing fingerprint scans.
* `test_my_thumb.py`: The real-world test script that uses OpenCV to convert a standard optical photo into a biometric scan and predicts the gender.

## How to Use
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure you have the trained weights `fingerprint_gender_model.pth` in the root directory.
3. To test your own thumb, take a picture, save it as `thumb.jpg`, and run `python test_my_thumb.py`.
