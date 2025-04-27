# Drone Classification System

A machine learning application that classifies drone images into four categories: DJI Inspire, DJI Mavic, DJI Phantom, and No Drone.

## Overview

This project provides a graphical user interface for processing drone images, training machine learning models, and classifying drone types. It implements and compares two different machine learning algorithms (Support Vector Machine and Random Forest Classifier) for drone classification.

## Features

- **Image Processing**: Automatically processes and prepares drone images for classification
- **Multiple ML Models**: Implements both SVM and Random Forest classifiers
- **Model Comparison**: Visually compares performance metrics between models
- **Interactive UI**: User-friendly interface for all operations
- **Performance Metrics**: Displays accuracy, precision, recall, and F-score for each model
- **Confusion Matrix**: Visual representation of model performance
- **Image Prediction**: Classifies new drone images using trained models

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - scikit-image
  - seaborn
  - tkinter
  - joblib

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy matplotlib scikit-learn scikit-image seaborn joblib
   ```
3. Organize your dataset in the following structure:
   ```
   Dataset/
     ├── dji_inspire/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── dji_mavic/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── dji_phantom/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     └── no_drone/
         ├── image1.jpg
         ├── image2.jpg
         └── ...
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. Follow these steps in the UI:
   - Click "Upload Dataset" to load your drone image dataset
   - Click "Image Processing" to process and prepare images
   - Click "Splitting" to split data into training and testing sets
   - Click "SVM_classifier" to train and evaluate the SVM model
   - Click "RFC Classifier" to train and evaluate the Random Forest model
   - Click "Compare Models" to visualize performance differences between models
   - Click "Prediction" to classify new drone images using the selected model

## Model Comparison

The application provides detailed comparison between SVM and RFC models:
- Tabular comparison of accuracy, precision, recall, and F-score
- Bar chart visualization of performance metrics
- Difference calculation between models

## Prediction

The prediction feature allows you to:
- Select an image file for classification
- Process the image automatically
- Display the prediction result
- Use either SVM or RFC model based on your selection

## Technical Details

- Images are resized to 150x150x3 pixels for processing
- Features are extracted from flattened image arrays
- Models are saved for future use without retraining
- Performance metrics are calculated using scikit-learn's evaluation metrics

## License

[Include your license information here] 