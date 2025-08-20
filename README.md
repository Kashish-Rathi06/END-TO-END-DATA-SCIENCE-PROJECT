# END-TO-END-DATA-SCIENCE-PROJECT

Company: CODTECH IT SOLUTIONS

NAME: Kashish Rathi

INTERN ID:CT06DH216G

DOMAIN:Data Science

DURACTION: 6 WEEKS

MENTOR: NEELA SANTOSH

---

### 📝 Script Description: CIFAR-10 Image Classification Using a CNN in TensorFlow

This Python script builds and trains a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify images from the **CIFAR-10 dataset**. CIFAR-10 is a popular benchmark dataset that contains 60,000 color images in 10 different classes such as airplanes, cars, birds, cats, and more. The dataset is divided into 50,000 training images and 10,000 test images, each of size 32x32 pixels with 3 color channels (RGB).

---

### 🔹 Step 1: Data Loading and Normalization

The function `load_data()` loads the CIFAR-10 dataset using TensorFlow's built-in API. It also normalizes the pixel values by dividing them by 255.0, converting the range from \[0, 255] to \[0, 1]. This normalization helps speed up training and improves model performance.

---

### 🔹 Step 2: CNN Model Construction

The `build_model()` function defines the CNN architecture using the `Sequential` API from Keras. The model consists of:

* Three **convolutional layers** with ReLU activation to extract features from the images.
* Two **max pooling layers** to reduce spatial dimensions and computation.
* A **flatten layer** to convert the 2D feature maps into a 1D vector.
* A **dense layer** with 64 neurons for further processing.
* A final **output layer** with 10 units and softmax activation to output class probabilities.

---

### 🔹 Step 3: Model Compilation and Training

The `train_model()` function compiles the model using the **Adam optimizer**, and **sparse categorical crossentropy** as the loss function, which is suitable for integer labels. The model is then trained for 10 epochs on the training set, with the test set used for validation. The training history, including loss and accuracy over epochs, is stored for visualization.

---

### 🔹 Step 4: Visualization of Training History

The `plot_history()` function uses Matplotlib to plot both training and validation accuracy and loss over each epoch. This helps in monitoring the model’s performance and identifying potential issues like overfitting.

---

### 🔹 Step 5: Making and Visualizing Predictions

Finally, the `show_predictions()` function selects 10 random test images, makes predictions using the trained model, and displays each image with its true and predicted labels. This offers a visual understanding of how well the model is performing.

---

### ✅ Conclusion

This script provides a full pipeline for training a CNN to classify images in the CIFAR-10 dataset. It includes loading and preprocessing data, building a deep learning model, training, evaluating, and visualizing predictions. It’s a great template for anyone learning deep learning and image classification using TensorFlow.

#OUTPUT

<img width="1915" height="1067" alt="Image" src="https://github.com/user-attachments/assets/4e16723f-5bcc-49d0-96e5-9d1678ef44e2" />
