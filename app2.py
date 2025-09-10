import streamlit as st
import time
from PIL import Image
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Machine Learning Dashboard - Iris Prediction",
    layout="wide",
    page_icon="🌸"
)

# =========================
# HEADER
# =========================
st.markdown("""
# 🌸 Welcome to My Machine Learning Dashboard

This dashboard created by : [Muhammad Alif Fahrezi](https://www.linkedin.com/in/aliffahrezii/)  
Use the sidebar to explore the available features.
---
""")

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.selectbox(
    "📂 Choose an option:",
    ("-","Iris Species Prediction")
)

# =========================
# CACHE MODEL
# =========================
@st.cache_resource
def load_model():
    with open("generate_iris.pkl", "rb") as file:
        return pickle.load(file)

# =========================
# FUNCTION: IRIS APP
# =========================
def iris_app():
    st.markdown("""
    ## 🔮 Iris Species Prediction  
    This app predicts the **Iris Species**  
    Dataset: [Iris dataset (UCIML)](https://www.kaggle.com/uciml/iris)  
    """)

    # Tabs
    tab1, tab2 = st.tabs(["🔮 Prediction", "📘 About Model"])

    with tab1:
        # Sidebar Input
        st.sidebar.header("📥 User Input Features")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload your input CSV file here"
        )

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        else:
            def manual_input():
                st.sidebar.subheader("🔧 Manual Input")
                sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.8)
                sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.0)
                petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.9, 4.3)
                petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.3)

                data = {
                    'SepalLengthCm': sepal_length,
                    'SepalWidthCm': sepal_width,
                    'PetalLengthCm': petal_length,
                    'PetalWidthCm': petal_width
                }
                return pd.DataFrame(data, index=[0])

            input_df = manual_input()

        # Display image
        try:
            img = Image.open("iris.JPG")
            st.image(img, width=500, caption="Sample Iris Flower")
        except:
            st.warning("⚠️ iris.JPG not found in folder. Please add the image.")

        # Show input table
        st.markdown("### 📊 Input Data")
        st.write(input_df)

        # Prediction Button
        if st.sidebar.button("🚀 Predict!"):
            with st.spinner("⏳ Running prediction..."):
                time.sleep(2)

                model = load_model()
                prediction = model.predict(input_df)

                # Mapping
                label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
                results = [label_map[p] for p in prediction]

                # Show single prediction
                if len(results) == 1:
                    st.markdown(f"""
                    <div style="padding:15px; background-color:#f0f0ff; border-radius:10px;">
                    🌸 <b>Prediction Result:</b> {results[0]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("🎉 Predictions completed for multiple rows!")
                    st.dataframe(pd.DataFrame({"Prediction": results}))

                    # Plot visualization
                    counts = pd.Series(prediction).value_counts().reindex([0, 1, 2], fill_value=0)
                    fig, ax = plt.subplots()
                    ax.bar(["Setosa", "Versicolor", "Virginica"], counts)
                    ax.set_ylabel("Count")
                    ax.set_title("Prediction Distribution")
                    st.pyplot(fig)

    with tab2:
        st.markdown("""
        ### 📘 About Model
        - **Algorithm**: Support Vector Machine (SVM) with GridSearchCV  
        - **Features**: Sepal Length, Sepal Width, Petal Length, Petal Width  
        - **Output Classes**:  
          - 🌱 Iris-setosa  
          - 🌿 Iris-versicolor  
          - 🌸 Iris-virginica  
        - Model trained and saved as `generate_iris.pkl`  
        """)

# =========================
# PAGE ROUTER
# =========================
if menu == "Iris Species Prediction":
    iris_app()





