
import pandas as pd
import numpy as np
import streamlit as st


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
    data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

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

@st.cache_resource
def train_model(data):
    feature_cols = [
        col for col in data.columns
        if col not in ['customerID', 'TotalCharges', 'Churn', 'Churn_Flag']
    ]

    X = data[feature_cols]
    y = data['Churn_Flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    cat_cols = X_train.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-score': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    }

    return model, feature_cols, metrics

#Interactive dashboard filters

df = load_data()
model, feature_cols, metrics = train_model(df)

st.sidebar.header('Interactive Filters')

contract_filter = st.sidebar.multiselect(
    'Filter by contract type',
    options=sorted(df['Contract'].unique()),
    default=sorted(df['Contract'].unique())
)

internet_filter = st.sidebar.multiselect(
    'Filter by internet service',
    options=sorted(df['InternetService'].unique()),
    default=sorted(df['InternetService'].unique())
)

charge_range = st.sidebar.slider(
    'Monthly charges range',
    min_value=float(df['MonthlyCharges'].min()),
    max_value=float(df['MonthlyCharges'].max()),
    value=(float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()))
)

search_customer = st.sidebar.text_input('Search customer ID')

show_only_churn = st.sidebar.checkbox('Show only actual churn customers')

filtered_df = df[
    (df['Contract'].isin(contract_filter)) &
    (df['InternetService'].isin(internet_filter)) &
    (df['MonthlyCharges'].between(charge_range[0], charge_range[1]))
].copy()

if search_customer:
    filtered_df = filtered_df[
        filtered_df['customerID'].str.contains(
            search_customer,
            case=False,
            na=False
        )
    ].copy()

if show_only_churn:
    filtered_df = filtered_df[
        filtered_df['Churn'] == 'Yes'
    ].copy()

if len(filtered_df) > 0:
    filtered_df['Predicted_Churn_Probability'] = model.predict_proba(
        filtered_df[feature_cols]
    )[:, 1]

    filtered_df['Risk_Level'] = pd.cut(
        filtered_df['Predicted_Churn_Probability'],
        bins=[0, 0.4, 0.7, 1],
        labels=['Low', 'Medium', 'High']
    )

    top_risk = filtered_df.sort_values(
        'Predicted_Churn_Probability',
        ascending=False
    )[[
        'customerID',
        'Contract',
        'InternetService',
        'tenure',
        'MonthlyCharges',
        'Churn',
        'Predicted_Churn_Probability',
        'Risk_Level'
    ]].head(10)

    st.subheader('Predictive Output: Top Customers at Risk of Churn')
    st.dataframe(top_risk, use_container_width=True)

else:
    st.warning('No customers match the selected filters.')
