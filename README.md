Here’s the updated README to match the visual style you want with bold headers, clear icons, and a modern layout similar to the reference image:

---

# **Handwritten Words Recognition Using CNN** ✍️🧠

![PYTHON](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![NUMPY](https://img.shields.io/badge/NUMPY-013243?style=for-the-badge&logo=numpy&logoColor=white) 
![JUPYTER](https://img.shields.io/badge/JUPYTER-F37626?style=for-the-badge&logo=jupyter&logoColor=white) 
![TENSORFLOW](https://img.shields.io/badge/TENSORFLOW-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![KERAS](https://img.shields.io/badge/KERAS-D00000?style=for-the-badge&logo=keras&logoColor=white)

---

## **Overview** 🎯
The **Handwritten Words Recognition** project focuses on detecting and classifying handwritten words using a **Convolutional Neural Network (CNN)**. The implementation includes **image preprocessing**, model training, and prediction.

---

## **Features** 🔧
- **Dataset Extraction:** Reads data from `.rar` files.
- **Image Processing:** Converts to grayscale, resizes to `64x64`, and normalizes images.
- **Model Training:** Utilizes CNN with dropout layers to prevent overfitting.
- **Prediction:** Outputs the label of handwritten words.

---

## **Prerequisites** 🛠️
Ensure the following libraries are installed:
```bash
pip install tensorflow opencv-python numpy scikit-learn rarfile
```

---

## **Dataset Structure** 📂
Place the dataset in a `.rar` file as follows:
```
data.rar
  |-- label1/
  |     |-- image1.png
  |-- label2/
        |-- image2.png
```

---

## **Steps to Run the Code** 🚀

1. **Extract Dataset**
   - Update `rar_path` and `extract_path` in the script.

2. **Preprocess Data**
   - Resizes images, converts to grayscale, and encodes labels.

3. **Train the Model**
   - Execute:
     ```python
     model.fit(X_train, y_train, 
               validation_data=(X_test, y_test),
               batch_size=32, epochs=20)
     ```

4. **Save the Model**
   - Model saved as `word_recognition_model.h5`.

5. **Make Predictions**
   - Predict using:
     ```python
     predicted_label = label_encoder.inverse_transform([np.argmax(model.predict(test_image[np.newaxis, ...]))])
     ```

---

## **Results** 📊

| Metric               | Value        |
|-----------------------|--------------|
| **Test Loss**        | `0.024`      |
| **Test Accuracy**    | `99.3%`      |

---

## **License** 📜  
This project is licensed under the **MIT License**.

---

## **Acknowledgments** 🙌
- TensorFlow 🧠
- Keras 💻
- OpenCV 🎨
- NumPy 📊

---

This README should now visually align with the style you're aiming for! Let me know if you'd like any other tweaks. 🚀
