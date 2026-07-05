import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(
    page_title='Telco Churn Decision Dashboard',
    layout='wide'
)

@st.cache_data
def load_data():
    data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    data['TotalCharges_numeric'] = pd.to_numeric(
        data['TotalCharges'],
        errors='coerce'
    ).fillna(0)

    data['ChargesPerTenure'] = (
        data['TotalCharges_numeric'] / (data['tenure'] + 1)
    )

    data['TenureGroup'] = pd.cut(
        data['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months']
    )

    data['Churn_Flag'] = data['Churn'].map({'No': 0, 'Yes': 1})

    return data
