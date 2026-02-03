# 🌦️ Prédiction des Précipitations à Dhaka

## 📋 Description du Projet

Ce projet utilise le **Machine Learning** pour prédire les précipitations à Dhaka, Bangladesh. Une application web interactive Streamlit permet de saisir les paramètres météorologiques et d'obtenir une prédiction en temps réel.

🔗 **Application déployée :** [https://ml-projet-oumaima.streamlit.app/](https://ml-projet-oumaima.streamlit.app/)

---

## 📊 Dataset

- **Source :** Données météorologiques de Dhaka
- **Fichier :** `datasets/dhaka_weather_data_full.csv`
- **Taille :** 7670 échantillons
- **Features :** 16 variables météorologiques

### Variables utilisées :

| Variable | Description | Unité |
|----------|-------------|-------|
| T2M | Température moyenne à 2m | °C |
| T2MDEW | Point de rosée | °C |
| T2MWET | Température humide | °C |
| TS | Température de surface | °C |
| T2M_RANGE | Amplitude thermique | °C |
| T2M_MAX | Température maximale | °C |
| T2M_MIN | Température minimale | °C |
| QV2M | Humidité spécifique | g/kg |
| RH2M | Humidité relative | % |
| PS | Pression atmosphérique | kPa |
| WS10M | Vitesse du vent | m/s |
| WS10M_MAX | Vitesse max du vent | m/s |
| WS10M_MIN | Vitesse min du vent | m/s |
| WS10M_RANGE | Amplitude vitesse vent | m/s |
| WD10M | Direction du vent | ° |
| Month | Mois de l'année | 1-12 |

---

## 🤖 Modèles de Machine Learning

Plusieurs modèles ont été entraînés et comparés :

| Modèle | Accuracy |
|--------|----------|
| 🏆 **Random Forest** | ~85% |
| Gradient Boosting | ~84% |
| ANN (MLP) | ~83% |
| AdaBoost | ~82% |
| KNN | ~80% |

Le modèle **Random Forest** avec 200 estimateurs a été sélectionné pour le déploiement.

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer l'application localement

```bash
cd ml_projet
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

---

## 📁 Structure du Projet

```
ml-projet/
├── requirements.txt          # Dépendances Python
├── README.md                  # Documentation
└── ml_projet/
    ├── app.py                 # Application Streamlit
    ├── Untitled10 (1).ipynb   # Notebook d'analyse et entraînement
    ├── datasets/
    │   └── dhaka_weather_data_full.csv
    └── models/
        ├── rain_prediction_model.pkl  # Modèle Random Forest
        ├── scaler.pkl                 # StandardScaler
        └── feature_names.pkl          # Liste des features
```

---

## 📓 Notebook

Le notebook `Untitled10 (1).ipynb` contient :

1. **Exploration des données** - Analyse statistique et visualisations
2. **Prétraitement** - Gestion des valeurs manquantes, normalisation
3. **Entraînement des modèles** - 5 algorithmes comparés
4. **Évaluation** - Métriques de performance
5. **Sauvegarde** - Export des modèles pour le déploiement

---

## 🛠️ Technologies Utilisées

- **Python** - Langage de programmation
- **Pandas & NumPy** - Manipulation des données
- **Scikit-learn** - Modèles de Machine Learning
- **Matplotlib & Seaborn** - Visualisations
- **Streamlit** - Interface web interactive
- **Joblib** - Sérialisation des modèles

---

## 👥 Auteurs

Projet de Machine Learning

---

## 📄 Licence

Ce projet est à des fins éducatives.
