import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Churn Prediction", layout="wide")

# -----------------------------
# White Theme Styling
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #FFFFFF;
    color: #1F2937;
}

/* Labels */
label, .stMarkdown, .stText {
    color: #1F2937 !important;
}

/* Headers */
h1, h2, h3 {
    color: #0E3B5F !important;
}

/* Form Container */
div[data-testid="stForm"] {
    background-color: #F9FAFB;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #E5E7EB;
}

/* Input Fields */
input, select, textarea {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

/* Stylish Button */
button[kind="formSubmit"] {
    background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    transition: 0.3s ease;
}

/* Hover Effect */
button[kind="formSubmit"]:hover {
    background: linear-gradient(90deg, #1D4ED8, #2563EB) !important;
    transform: scale(1.05);
}

/* Result Box */
.result-box {
    background-color: #F3F4F6;
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    border-left: 5px solid #3B82F6;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, threshold


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_input(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    bins = [0, 3, 6, 12, 24, 48, 72]
    labels = [
        '0-3 Months', '4-6 Months', '7-12 Months',
        '13-24 Months', '25-48 Months', '49+ Months'
    ]

    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    df = df.drop(columns=['customerID', 'tenure'], errors='ignore')

    return df


# -----------------------------
# Main App
# -----------------------------
def main():

    st.markdown("<h1 style='text-align: center;'>📊 Telecom Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    model, threshold = load_model()

    with st.form("input_form"):

        user_input = {}

        # -----------------------------
        # Customer Info
        # -----------------------------
        st.subheader("👤 Customer Info")

        col1, col2 = st.columns(2)

        with col1:
            user_input['gender'] = st.selectbox("Gender", ['Female', 'Male'])

            senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
            user_input['SeniorCitizen'] = 1 if senior == 'Yes' else 0

        with col2:
            user_input['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
            user_input['Dependents'] = st.selectbox("Dependents", ['No', 'Yes'])


        # -----------------------------
        # Billing
        # -----------------------------
        st.subheader("💳 Billing")

        col3, col4, col5 = st.columns(3)

        with col3:
            user_input['Contract'] = st.selectbox(
                "Contract", ['Month-to-month', 'One year', 'Two year']
            )

        with col4:
            user_input['PaymentMethod'] = st.selectbox(
                "Payment Method",
                [
                    'Electronic check',
                    'Mailed check',
                    'Bank transfer (automatic)',
                    'Credit card (automatic)'
                ]
            )

        with col5:
            user_input['PaperlessBilling'] = st.selectbox(
                "Paperless Billing", ['Yes', 'No']
            )


        # -----------------------------
        # Services
        # -----------------------------
        st.subheader("📡 Services")

        col6, col7 = st.columns(2)

        with col6:
            user_input['PhoneService'] = st.selectbox("Phone Service", ['No', 'Yes'])

            user_input['MultipleLines'] = st.selectbox(
                "Multiple Lines", ['No phone service', 'No', 'Yes']
            )

            user_input['InternetService'] = st.selectbox(
                "Internet Service", ['DSL', 'Fiber optic', 'No']
            )

            user_input['StreamingTV'] = st.selectbox(
                "Streaming TV", ['No', 'Yes', 'No internet service']
            )

        with col7:
            user_input['OnlineSecurity'] = st.selectbox(
                "Online Security", ['No', 'Yes', 'No internet service']
            )

            user_input['OnlineBackup'] = st.selectbox(
                "Online Backup", ['Yes', 'No', 'No internet service']
            )

            user_input['DeviceProtection'] = st.selectbox(
                "Device Protection", ['No', 'Yes', 'No internet service']
            )

            user_input['TechSupport'] = st.selectbox(
                "Tech Support", ['No', 'Yes', 'No internet service']
            )

            user_input['StreamingMovies'] = st.selectbox(
                "Streaming Movies", ['No', 'Yes', 'No internet service']
            )


        # -----------------------------
        # Usage
        # -----------------------------
        st.subheader("📊 Usage Details")

        col8, col9, col10 = st.columns(3)

        with col8:
            user_input['tenure'] = st.slider("Tenure (months)", 0, 72, 12)

        with col9:
            user_input['MonthlyCharges'] = st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0
            )

        with col10:
            user_input['TotalCharges'] = st.number_input(
                "Total Charges", min_value=0.0, value=1000.0
            )


        # Submit Button
        submitted = st.form_submit_button("🚀 Predict Churn")


    # -----------------------------
    # Prediction Result
    # -----------------------------
    if submitted:

        input_df = pd.DataFrame([user_input])
        input_df = preprocess_input(input_df)

        prob = model.predict_proba(input_df)[:, 1][0]
        pred = int(prob > threshold)

        st.markdown("## 📈 Result")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        colA, colB = st.columns(2)

        with colA:
            st.markdown(
                f"<h3>Churn Probability: {prob:.2f}</h3>",
                unsafe_allow_html=True
            )

        with colB:
            st.markdown(
                f"<h3>Threshold: {threshold:.2f}</h3>",
                unsafe_allow_html=True
            )

        if pred == 1:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")

        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    main()