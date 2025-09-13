import streamlit as st
import pandas as pd
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from scipy import stats
from sklearn.decomposition import PCA
import datetime  # ✅ Added for timestamp in signature

def DataExploration():
    st.markdown("""<h1 style='color:#87CEEB;'>Data Exploration </h1>""", unsafe_allow_html=True)
    
    st.write("Upload a dataset and explore its basic properties.")

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type")
            return

        st.subheader("Dataset Overview")
        st.write("### First Few Rows")
        st.dataframe(df.head())

        st.write("### Data Types and Missing Values")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("### Summary Statistics")
        st.dataframe(df.describe(include="all"))

        st.write("### Missing Values")
        st.dataframe(df.isnull().sum())

        st.write("### Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.write("### Column Names")
        st.write(df.columns.tolist())

        if st.checkbox("Fill Missing Values with Mean/Mode/Median"):
            fill_method = st.radio("Select Method", ("Mean", "Median", "Mode"))
            for col in df.select_dtypes(include=[np.number]).columns:
                if fill_method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill_method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif fill_method == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values handled successfully.")

        if st.checkbox("Remove Duplicate Rows"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicate rows removed successfully.")

        if st.checkbox("Apply Feature Scaling"):
            scaler_option = st.radio("Select Scaling Method", ("StandardScaler", "MinMaxScaler", "RobustScaler"))
            scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler() if scaler_option == "MinMaxScaler" else RobustScaler()
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.success("Feature scaling applied successfully.")

        if st.checkbox("Apply Label Encoding to Categorical Features"):
            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col])
            st.success("Categorical encoding applied successfully.")

        if st.checkbox("Remove Outliers using Z-Score"):
            threshold = st.slider("Select Z-Score threshold", 1.0, 5.0, 3.0)
            df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < threshold).all(axis=1)]
            st.success("Outliers removed successfully.")

        if st.checkbox("Remove Low Variance Features"):
            variance_threshold = st.slider("Select Variance Threshold", 0.0, 0.1, 0.01)
            low_variance_cols = df.var()[df.var() < variance_threshold].index.tolist()
            df.drop(columns=low_variance_cols, inplace=True)
            st.success("Low variance features removed successfully.")

        if st.checkbox("Apply PCA for Dimensionality Reduction"):
            n_components = st.slider("Select Number of Components", 1, min(df.shape[1], 5), 2)
            pca = PCA(n_components=n_components)
            df_pca = pca.fit_transform(df.select_dtypes(include=[np.number]))
            df = pd.DataFrame(df_pca, columns=[f"PC{i+1}" for i in range(n_components)])
            st.success("PCA applied successfully.")

        if st.checkbox("Extract Features from Date-Time Columns"):
            datetime_cols = df.select_dtypes(include=["datetime", "object"]).columns
            selected_col = st.selectbox("Select Date-Time Column", datetime_cols)
            df[selected_col] = pd.to_datetime(df[selected_col])
            df[f"{selected_col}_year"] = df[selected_col].dt.year
            df[f"{selected_col}_month"] = df[selected_col].dt.month
            df[f"{selected_col}_day"] = df[selected_col].dt.day
            df.drop(columns=[selected_col], inplace=True)
            st.success("Date-time features extracted successfully.")

        # ✅ Add Signature Column Before Download
        signature = f"Processed_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        df["__processed_signature__"] = signature

        st.write("### Download Processed Dataset")
        processed_file = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", processed_file, "processed_dataset.csv", "text/csv")
    else:
        st.info("Please upload a file to begin.")

if __name__ == "__main__":
    DataExploration()
