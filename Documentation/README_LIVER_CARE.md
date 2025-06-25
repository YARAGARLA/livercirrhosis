# 🩺 Liver Care - Cirrhosis Stage Prediction Web App

**Liver Care** is a machine learning-powered Flask web application that predicts the **stage of liver cirrhosis** based on user inputs.

## 🚀 Features
- User-friendly web form
- Predicts cirrhosis stage using trained ML model
- Categorical encoding + L1 normalization
- Clean frontend with HTML/CSS

## 🧠 ML Model
- **Model**: Random Forest (GridSearchCV tuned)
- **Target**: Stage (0 = No Cirrhosis, 1 = Cirrhosis)

## 📁 Structure
```
LIVER_CARE/
├── app.py
├── normalizer.pkl
├── random_forest_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── css/style.css
```

## 🧪 Run Locally
```bash
pip install flask pandas scikit-learn joblib
python app.py
```

Open http://localhost:5000

## 👩‍💻 Author
**Vaishnavi Vuppala**