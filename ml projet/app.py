import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="🌧️ Prédiction de Précipitation - Dhaka",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer le design
st.markdown("""
<style>
    /* Titre principal */
    .main-title {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .main-title h1 {
        color: #00d4ff;
        font-size: 2.5em;
        margin: 0;
    }
    
    .main-title p {
        color: #a0a0a0;
        font-size: 1.2em;
    }
    
    /* Résultat pluie */
    .rain-result {
        background: linear-gradient(145deg, #1e3a5f, #2a4a70);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 3px solid #4a90d9;
        box-shadow: 0 8px 25px rgba(74, 144, 217, 0.3);
    }
    
    /* Résultat pas de pluie */
    .no-rain-result {
        background: linear-gradient(145deg, #2d5a1f, #3d6a2f);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 3px solid #7cb342;
        box-shadow: 0 8px 25px rgba(124, 179, 66, 0.3);
    }
    
    /* Probabilités */
    .prob-card {
        background: linear-gradient(145deg, #252538, #2d2d45);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Bouton */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        color: white;
        font-size: 1.3em;
        font-weight: bold;
        padding: 15px 30px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.6);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(145deg, #252538, #2d2d45);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-title">
    <h1>🌦️ Prédiction de Précipitation à Dhaka</h1>
    <p>🤖 Intelligence Artificielle pour la Prédiction Météorologique</p>
</div>
""", unsafe_allow_html=True)

# Chargement du modèle et du scaler
@st.cache_resource
def load_model():
    model = joblib.load('models/rain_prediction_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

# Vérifier si les fichiers existent
if not os.path.exists('models/rain_prediction_model.pkl'):
    st.error("⚠️ Le modèle n'a pas été trouvé. Veuillez d'abord exécuter le notebook pour entraîner et sauvegarder le modèle.")
    st.stop()

model, scaler, feature_names = load_model()

# Configuration des features avec valeurs réelles du dataset Dhaka
feature_config = {
    'T2M': {
        'label': 'T2M - Température Moyenne à 2m',
        'unit': '°C',
        'min': 10.0,
        'max': 35.0,
        'default': 25.0,
        'step': 0.5,
        'icon': '🌡️',
        'help': 'Température moyenne de l\'air à 2 mètres du sol'
    },
    'T2MDEW': {
        'label': 'T2MDEW - Point de Rosée',
        'unit': '°C',
        'min': 0.0,
        'max': 30.0,
        'default': 18.0,
        'step': 0.5,
        'icon': '💧',
        'help': 'Température à laquelle l\'air devient saturé en vapeur d\'eau'
    },
    'T2MWET': {
        'label': 'T2MWET - Température Humide',
        'unit': '°C',
        'min': 5.0,
        'max': 32.0,
        'default': 20.0,
        'step': 0.5,
        'icon': '🌡️',
        'help': 'Température mesurée avec un thermomètre à bulbe humide'
    },
    'TS': {
        'label': 'TS - Température de Surface',
        'unit': '°C',
        'min': 10.0,
        'max': 45.0,
        'default': 26.0,
        'step': 0.5,
        'icon': '🌍',
        'help': 'Température à la surface du sol'
    },
    'T2M_RANGE': {
        'label': 'T2M_RANGE - Amplitude Thermique',
        'unit': '°C',
        'min': 3.0,
        'max': 20.0,
        'default': 12.0,
        'step': 0.5,
        'icon': '📊',
        'help': 'Différence entre température max et min de la journée'
    },
    'T2M_MAX': {
        'label': 'T2M_MAX - Température Maximale',
        'unit': '°C',
        'min': 15.0,
        'max': 45.0,
        'default': 32.0,
        'step': 0.5,
        'icon': '🔥',
        'help': 'Température maximale de la journée'
    },
    'T2M_MIN': {
        'label': 'T2M_MIN - Température Minimale',
        'unit': '°C',
        'min': 5.0,
        'max': 30.0,
        'default': 18.0,
        'step': 0.5,
        'icon': '❄️',
        'help': 'Température minimale de la journée'
    },
    'QV2M': {
        'label': 'QV2M - Humidité Spécifique',
        'unit': 'g/kg',
        'min': 3.0,
        'max': 25.0,
        'default': 15.0,
        'step': 0.5,
        'icon': '💨',
        'help': 'Quantité de vapeur d\'eau dans l\'air (g/kg)'
    },
    'RH2M': {
        'label': 'RH2M - Humidité Relative',
        'unit': '%',
        'min': 30.0,
        'max': 100.0,
        'default': 70.0,
        'step': 1.0,
        'icon': '💦',
        'help': 'Pourcentage d\'humidité dans l\'air'
    },
    'PS': {
        'label': 'PS - Pression Atmosphérique',
        'unit': 'kPa',
        'min': 98.0,
        'max': 103.0,
        'default': 101.0,
        'step': 0.1,
        'icon': '🌀',
        'help': 'Pression atmosphérique à la surface'
    },
    'WS10M_RANGE': {
        'label': 'WS10M_RANGE - Amplitude Vitesse Vent',
        'unit': 'm/s',
        'min': 0.5,
        'max': 15.0,
        'default': 3.0,
        'step': 0.1,
        'icon': '🌬️',
        'help': 'Différence entre vitesse max et min du vent'
    },
    'WS10M': {
        'label': 'WS10M - Vitesse du Vent',
        'unit': 'm/s',
        'min': 0.5,
        'max': 12.0,
        'default': 2.5,
        'step': 0.1,
        'icon': '💨',
        'help': 'Vitesse moyenne du vent à 10 mètres'
    },
    'WD10M': {
        'label': 'WD10M - Direction du Vent',
        'unit': '°',
        'min': 0.0,
        'max': 360.0,
        'default': 180.0,
        'step': 5.0,
        'icon': '🧭',
        'help': 'Direction du vent (0°=Nord, 90°=Est, 180°=Sud, 270°=Ouest)'
    },
    'WS10M_MAX': {
        'label': 'WS10M_MAX - Vitesse Max du Vent',
        'unit': 'm/s',
        'min': 1.0,
        'max': 20.0,
        'default': 4.0,
        'step': 0.1,
        'icon': '🌪️',
        'help': 'Vitesse maximale du vent de la journée'
    },
    'WS10M_MIN': {
        'label': 'WS10M_MIN - Vitesse Min du Vent',
        'unit': 'm/s',
        'min': 0.0,
        'max': 8.0,
        'default': 1.0,
        'step': 0.1,
        'icon': '🍃',
        'help': 'Vitesse minimale du vent de la journée'
    },
    'Month': {
        'label': 'Month - Mois de l\'année',
        'unit': '',
        'min': 1,
        'max': 12,
        'default': 6,
        'step': 1,
        'icon': '📅',
        'help': '1=Janvier, 6=Juin, 12=Décembre'
    }
}

# Sidebar pour les entrées
st.sidebar.markdown("## 📊 Paramètres Météorologiques")
st.sidebar.markdown("---")

# Création des inputs
input_values = {}

# Section Température
st.sidebar.markdown("### 🌡️ Température")
temp_features = ['T2M', 'T2MDEW', 'T2MWET', 'TS', 'T2M_RANGE', 'T2M_MAX', 'T2M_MIN']
for feat in temp_features:
    if feat in feature_names and feat in feature_config:
        config = feature_config[feat]
        input_values[feat] = st.sidebar.slider(
            f"{config['icon']} {config['label']} ({config['unit']})",
            min_value=float(config['min']),
            max_value=float(config['max']),
            value=float(config['default']),
            step=float(config['step']),
            help=config['help']
        )

st.sidebar.markdown("---")

# Section Humidité
st.sidebar.markdown("### 💧 Humidité")
humidity_features = ['QV2M', 'RH2M']
for feat in humidity_features:
    if feat in feature_names and feat in feature_config:
        config = feature_config[feat]
        input_values[feat] = st.sidebar.slider(
            f"{config['icon']} {config['label']} ({config['unit']})",
            min_value=float(config['min']),
            max_value=float(config['max']),
            value=float(config['default']),
            step=float(config['step']),
            help=config['help']
        )

st.sidebar.markdown("---")

# Section Pression
st.sidebar.markdown("### 🌀 Pression")
if 'PS' in feature_names and 'PS' in feature_config:
    config = feature_config['PS']
    input_values['PS'] = st.sidebar.slider(
        f"{config['icon']} {config['label']} ({config['unit']})",
        min_value=float(config['min']),
        max_value=float(config['max']),
        value=float(config['default']),
        step=float(config['step']),
        help=config['help']
    )

st.sidebar.markdown("---")

# Section Vent
st.sidebar.markdown("### 🌬️ Vent")
wind_features = ['WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WD10M']
for feat in wind_features:
    if feat in feature_names and feat in feature_config:
        config = feature_config[feat]
        input_values[feat] = st.sidebar.slider(
            f"{config['icon']} {config['label']} ({config['unit']})",
            min_value=float(config['min']),
            max_value=float(config['max']),
            value=float(config['default']),
            step=float(config['step']),
            help=config['help']
        )

st.sidebar.markdown("---")

# Section Mois
st.sidebar.markdown("### 📅 Période")
if 'Month' in feature_names and 'Month' in feature_config:
    mois_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                  'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    selected_month = st.sidebar.selectbox(
        "📅 Mois",
        options=list(range(1, 13)),
        format_func=lambda x: f"{x} - {mois_names[x-1]}",
        index=5,
        help="Sélectionnez le mois de l'année"
    )
    input_values['Month'] = selected_month

# Corps principal
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### 📋 Récapitulatif des Valeurs")
    
    # Créer un dataframe pour l'affichage
    display_data = []
    for feat in feature_names:
        if feat in feature_config:
            config = feature_config[feat]
            value = input_values.get(feat, config['default'])
            label_short = config['label'].split(' - ')[1] if ' - ' in config['label'] else config['label']
            display_data.append({
                'Feature': f"{config['icon']} {feat}",
                'Description': label_short,
                'Valeur': f"{value} {config['unit']}"
            })
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=400)

with col2:
    st.markdown("### 🔮 Prédiction")
    st.markdown("")
    
    # Bouton de prédiction centré
    predict_button = st.button("🔮 PRÉDIRE LA MÉTÉO", type="primary", use_container_width=True)
    
    st.markdown("")
    
    if predict_button:
        # Préparer les données pour la prédiction
        input_array = np.array([[input_values.get(f, 0) for f in feature_names]])
        
        # Normaliser les données
        input_scaled = scaler.transform(input_array)
        
        # Faire la prédiction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        prob_no_rain = prediction_proba[0] * 100
        prob_rain = prediction_proba[1] * 100
        
        st.markdown("---")
        
        # Afficher le résultat
        if prediction == 1:
            st.markdown(f"""
            <div class="rain-result">
                <h1 style="color: #4a90d9; margin: 0; font-size: 3em;">🌧️</h1>
                <h2 style="color: #4a90d9; margin: 10px 0;">PRÉCIPITATION PRÉVUE</h2>
                <p style="color: #87CEEB; font-size: 18px;">Des précipitations sont attendues aujourd'hui</p>
                <h3 style="color: #00d4ff; margin-top: 20px;">{prob_rain:.1f}% de probabilité</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-rain-result">
                <h1 style="color: #7cb342; margin: 0; font-size: 3em;">☀️</h1>
                <h2 style="color: #7cb342; margin: 10px 0;">TEMPS SEC</h2>
                <p style="color: #c5e1a5; font-size: 18px;">Pas de précipitation prévue aujourd'hui</p>
                <h3 style="color: #aed581; margin-top: 20px;">{prob_no_rain:.1f}% de probabilité</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("### 📊 Détail des Probabilités")
        
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            st.markdown(f"""
            <div class="prob-card">
                <h2 style="color: #7cb342; margin: 0;">☀️ Pas de précipitation</h2>
                <h1 style="color: #aed581; margin: 10px 0;">{prob_no_rain:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col_p2:
            st.markdown(f"""
            <div class="prob-card">
                <h2 style="color: #4a90d9; margin: 0;">🌧️ Précipitation</h2>
                <h1 style="color: #87CEEB; margin: 10px 0;">{prob_rain:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Barre de progression
        st.markdown("")
        st.markdown("**Probabilité de précipitation :**")
        st.progress(prob_rain / 100)
        
    else:
        st.markdown("""
        <div class="info-box">
            <h2 style="color: #888;">👆 Ajustez les paramètres</h2>
            <p style="color: #666;">Utilisez la barre latérale pour définir les conditions météorologiques, puis cliquez sur <b>PRÉDIRE</b></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1a1a2e, #16213e); border-radius: 10px;">
    <p style="color: #00d4ff; font-size: 1.1em; margin: 0;">🤖 Modèle : Random Forest Classifier | 📍 Données : Dhaka, Bangladesh</p>
    <p style="color: #888; margin-top: 10px;">Projet de Machine Learning - Prédiction Météorologique</p>
</div>
""", unsafe_allow_html=True)
