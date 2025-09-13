import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Home():
    # Custom Styling
    st.markdown("""
        <style>
            .main-title { text-align: center; color: #4A90E2; font-size: 36px; font-weight: bold; }
            .sub-title { text-align: center; color: #8A4AF3; font-size: 28px; font-weight: bold; }
            .section-title { color: #87CEEB; font-size: 24px; font-weight: bold; margin-top: 20px; }
            .content-text { font-size: 18px; color: #333333; line-height: 1.6; }
            .code-box { background-color: #1B1B2F; color: #FFFFFF; padding: 10px; border-radius: 5px; }
            .highlight { color: #17A2B8; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("<h1 class='main-title'>SmartPrep Hub -Transforming Raw Data into ML-Ready Gold </h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-title'>Hello, I am <span class='highlight'>Prathamesh Jadhav</span> 👋</h2>", unsafe_allow_html=True)
    st.write("""
    Data preprocessing is a crucial step in any machine learning pipeline. Raw data often contains inconsistencies such as 
    missing values, duplicates, and outliers. If we don’t clean and prepare the data properly, our models might give 
    inaccurate predictions.
    """)
    
    sections = {
        "1. Data Loading": "df = pd.read_csv('file.csv')",
        "2. Data Exploration": """
            df.head()  # Shows the first few rows
            df.info()  # Displays column data types and missing values
            df.describe()  # Provides statistical summary
        """,
        "3. Handling Missing Values": """
            df.dropna()  # Removes missing values
            df.fillna(df.mean())  # Fills missing values with mean
        """,
        "4. Handling Duplicates": "df.drop_duplicates()",
        "5. Data Type Conversion": "df['column'] = df['column'].astype(float)",
        "6. Handling Categorical Variables": """
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['category'] = le.fit_transform(df['category'])
            pd.get_dummies(df['category'])  # One-hot encoding
        """,
        "7. Feature Scaling/Normalization": """
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df[['column']])
        """,
        "8. Handling Outliers": """
            from scipy import stats
            df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]  # Removes outliers based on Z-score
        """,
        "9. Feature Engineering": """
            df['new_feature'] = df['feature1'] * df['feature2']  # Creating a new feature
        """,
        "10. Data Balancing": """
            from imblearn.over_sampling import SMOTE
            smote = SMOTE()
            X_resampled, y_resampled = smote.fit_resample(X, y)  # Handling imbalanced data
        """,
        "11. Removing Constant Features": "df = df.loc[:, df.nunique() > 1]",
        "12. Handling DateTime Features": "df['year'] = pd.to_datetime(df['date']).dt.year",
        "13. Binning Continuous Data": "df['binned'] = pd.cut(df['value'], bins=5, labels=False)",
        "14. Log Transformation": "df['log_value'] = np.log1p(df['value'])",
        "15. Power Transformation": """
            from sklearn.preprocessing import PowerTransformer
            pt = PowerTransformer()
            df[['column']] = pt.fit_transform(df[['column']])
        """,
        "16. Polynomial Features": """
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2)
            df_poly = poly.fit_transform(df[['feature1', 'feature2']])
        """,
        "17. Encoding Ordinal Variables": """
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder()
            df[['category']] = encoder.fit_transform(df[['category']])
        """,
        "18. Handling Skewness": "df['normalized'] = np.sqrt(df['skewed_column'])",
        "19. Filling Missing Categorical Data": "df['category'].fillna(df['category'].mode()[0], inplace=True)",
        "20. Removing High Correlation Features": "df = df.drop(columns=['highly_correlated_feature'])"
    }
    
    for title, code in sections.items():
        st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
        st.code(code, language='python')
    
    # Conclusion
    st.markdown("<h2 class='section-title'>Conclusion</h2>", unsafe_allow_html=True)
    st.write("""
    These preprocessing steps are essential for building accurate machine learning models. Properly cleaned and prepared data
    leads to better predictions and generalization.
    """)