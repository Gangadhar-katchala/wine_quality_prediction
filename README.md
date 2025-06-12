# Wine Quality Prediction

A machine learning project that predicts wine quality based on various physicochemical properties using Random Forest Classifier.

## 🍷 Features

- Predicts wine quality based on 12 input parameters
- Interactive web interface (Flask)
- Data preprocessing and feature engineering
- Model persistence and logging

## 📋 Input Parameters

1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Color (red/white)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Gangadhar-katchala/wine_quality_prediction.git
    cd wine_quality_prediction
    ```

2. Create and activate virtual environment:
    ```sh
    python -m venv wine_quality
    wine_quality\Scripts\activate   # Windows
    # or
    source wine_quality/bin/activate  # Mac/Linux
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the application:
    ```sh
    python app.py
    ```

5. Open: `http://127.0.0.1:5000/predictdata`

## 📊 Model Performance

**Random Forest Classifier Results:**

| Class   | Precision | Recall | F1-score |
|---------|-----------|--------|----------|
| high    | 0.78      | 0.59   | 0.67     |
| low     | 0.83      | 0.10   | 0.18     |
| medium  | 0.87      | 0.96   | 0.91     |

- **Accuracy:** 0.85
- **Macro avg:** Precision 0.83, Recall 0.55, F1-score 0.59
- **Weighted avg:** Precision 0.85, Recall 0.85, F1-score 0.84

## 👥 Author

Gangadhar Katchala - [Gangadhar-katchala](https://github.com/Gangadhar-katchala) - katchalagangadhar@gmail.com

## 📄 License

This project is for educational purposes.

## 📞 Contact

Gangadhar Katchala - [@Gangadhar-katchala](https://twitter.com/Gangadhar-katchala) - katchalagangadhar@gmail.com

Project Link: [https://github.com/Gangadhar-katchala/wine_quality_prediction](https://github.com/Gangadhar-katchala/wine_quality_prediction)
