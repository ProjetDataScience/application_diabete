import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_csv("diabetes.csv")

# Séparer les features et la cible
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train[['Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] = scaler.fit_transform(X_train[['Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']])
X_test[['Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] = scaler.transform(X_test[['Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']])

# Entraîner le modèle
model = RandomForestClassifier(random_state=12345, max_depth=8, max_features=7, min_samples_split=2, n_estimators=500)
model.fit(X_train, y_train)

# Sauvegarder le modèle et le scaler
joblib.dump(model, 'model/diabete_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
