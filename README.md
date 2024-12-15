
# CNN-Based Melanoma Detection

## Project Overview
This project aims to build a custom convolutional neural network (CNN) to accurately detect melanoma, a severe type of skin cancer accounting for 75% of skin cancer-related deaths. Early detection through automated solutions can significantly reduce the manual effort required in diagnosis.

### Dataset Description
The dataset used in this project comprises 2357 images sourced from the International Skin Imaging Collaboration (ISIC). It includes malignant and benign oncological diseases, categorized as follows:
- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

### Dataset Preparation
- **Image Resizing**: All images are resized to 180x180 pixels.
- **Normalization**: Pixel values are scaled to the range [0,1].
- **Class Balancing**: Addressed class imbalances using the Augmentor library.

## Project Pipeline
1. **Data Understanding and Visualization**:
   - Visualize samples from each class.
2. **Model Building**:
   - A custom CNN model is created without pre-trained models.
   - The architecture consists of convolutional, pooling, and dense layers.
   - Overfitting is controlled using dropout layers.
3. **Model Training**:
   - The model is trained for ~20 epochs using appropriate loss functions and optimizers.
   - Class imbalances are rectified, and the model is retrained for ~30 epochs.
4. **Evaluation**:
   - The model's accuracy and loss curves are analyzed for signs of underfitting or overfitting.

## CNN Architecture
- **Input Shape**: (180, 180, 3)
- **Convolutional Layers**:
  - Three layers with filters: 32 → 64 → 128.
  - Followed by MaxPooling2D to reduce spatial dimensions.
- **Dense Layers**:
  - Flatten → Dense (128 neurons, ReLU) → Dense (9 neurons, Softmax).
- **Regularization**:
  - Dropout: 50% after convolutional layers, 25% after the first dense layer.
- **Optimization**:
  - Adam optimizer and categorical cross-entropy loss function.

## Key Findings
1. The training accuracy steadily increases, converging with validation accuracy around epoch 10.
2. Slight overfitting is observed after epoch 20, indicated by a divergence in training and validation loss curves.
3. Class imbalances initially hindered performance but were effectively addressed using augmentation techniques.

### Recommendations
- Implement **Batch Normalization** for faster convergence.
- Use **Early Stopping** to prevent overfitting.
- Fine-tune hyperparameters, such as learning rate and batch size.

## Visuals
### Sample Visualizations
(Add visuals of dataset classes and model results here.)

### Training and Validation Curves
(Add plots of accuracy and loss here.)

## Tools and Libraries Used
- TensorFlow, Keras
- NumPy, Matplotlib, Pandas
- Augmentor library for data augmentation
- Jupyter Notebook for implementation

## Instructions to Run
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook `Vaibhav_Aggarwal_NN.ipynb` for step-by-step implementation.

## References
- Dataset: [International Skin Imaging Collaboration (ISIC)](https://isic-archive.com/)
- TensorFlow Documentation

---

### Appendix: Additional Insights from Notebook
Key hyperparameters, model architecture visualization, and code snippets are included in the attached notebook.

