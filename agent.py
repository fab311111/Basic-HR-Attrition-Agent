import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
from io import BytesIO
import base64

# Load model
model = joblib.load("attrition_model.pkl")

st.set_page_config(page_title="AI Attrition Prediction Agent", layout="wide")

st.title("💼 HR Attrition Prediction AI Agent")
st.write("Enter employee details to predict attrition risk and get targeted HR recommendations.")

# Sidebar inputs
st.sidebar.header("Employee Profile")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
worklife_balance = st.sidebar.slider("Work-Life Balance (1-4)", 1, 4, 3)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)

# Convert to model input format
input_data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [monthly_income],
    'OverTime': [1 if overtime == "Yes" else 0],
    'JobSatisfaction': [job_satisfaction],
    'WorkLifeBalance': [worklife_balance],
    'YearsAtCompany': [years_at_company]
})

# Prediction button
if st.button("Predict Attrition Risk"):
    # Model prediction (probability)
    pred_prob = model.predict_proba(input_data)[0][1]  # probability of attrition
    risk = "High" if pred_prob > 0.65 else "Medium" if pred_prob > 0.4 else "Low"

    st.subheader(f"🧭 Predicted Attrition Risk: **{risk}** ({pred_prob:.2f} probability)")

    # HR recommendations
    if risk == "High":
        st.warning("High Risk: Employee likely to leave. Recommended Actions:")
        st.markdown("- Reduce overtime and improve work-life balance.")
        st.markdown("- Offer recognition or financial incentives.")
        st.markdown("- Provide career growth opportunities.")
    elif risk == "Medium":
        st.info("Medium Risk: Moderate likelihood of leaving. Recommended Actions:")
        st.markdown("- Schedule periodic engagement meetings.")
        st.markdown("- Monitor job satisfaction closely.")
        st.markdown("- Review compensation if below market average.")
    else:
        st.success("Low Risk: Employee stable. Recommended Actions:")
        st.markdown("- Maintain positive workplace environment.")
        st.markdown("- Continue performance recognition programs.")

    # Visualize probability
    st.progress(int(pred_prob * 100))

            # --- PDF Report Generation ---
        if st.button("📄 Generate PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Attrition Prediction Report", ln=True, align="C")

            pdf.set_font("Arial", "", 12)
            pdf.ln(10)
            pdf.multi_cell(0, 10, f"""
Employee Profile:
- Age: {age}
- Monthly Income: ${monthly_income}
- OverTime: {overtime}
- Job Satisfaction: {job_satisfaction}
- Work-Life Balance: {worklife_balance}
- Years at Company: {years_at_company}

Prediction:
- Risk Level: {risk}
- Probability: {prob:.2f}

Recommendations:
{chr(10).join(['- ' + rec for rec in recs])}
            """)

            # Save PDF in memory
            buffer = BytesIO()
            pdf.output(buffer)
            buffer.seek(0)

            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="Attrition_Report.pdf">📥 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.info("Please fill in employee details and click 'Predict Attrition Risk'.")

st.caption("Developed by [Tasleem Raheel] — HR Attrition Project (Streamlit Cloud)")

