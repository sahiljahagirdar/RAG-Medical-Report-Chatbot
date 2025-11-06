import sys
import streamlit as st
import pandas as pd
import re
import os
from Retrieval_pipeline import generate_explanation
import traceback
import time

sys.stdout.reconfigure(encoding="utf-8", errors="ignore")

st.set_page_config(page_title="🩺 Healthcare Report Explainer", layout="wide")
st.title("🩺 Healthcare Report Explainer")
st.write("Upload your medical report (PDF) to get easy explanations for each test.")

uploaded_file = st.file_uploader("Upload PDF Report", type=["pdf"])

if uploaded_file:
    os.makedirs("reports", exist_ok=True)
    file_path = os.path.join("reports", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("File uploaded successfully!")

    progress_text = "🔍 Analyzing your report... Please wait."
    progress_bar = st.progress(0, text=progress_text)

    for percent_complete in range(0, 80, 10):
        time.sleep(0.3)
        progress_bar.progress(percent_complete + 10, text=progress_text)

    try:
        explanation_text = generate_explanation(file_path)
    except ValueError as e:
        progress_bar.empty()
        if str(e) == "NOT_MEDICAL_REPORT":
            st.warning("⚠️ This file doesn’t look like a medical report. Please upload a valid lab report PDF.")
        else:
            st.error("⚠️ Something went wrong while generating the explanation.")
            st.text_area("Error details:", traceback.format_exc(), height=200)
        st.stop()
    except Exception:
        progress_bar.empty()
        st.error("⚠️ Unexpected error occurred.")
        st.text_area("Error details:", traceback.format_exc(), height=200)
        st.stop()

    progress_bar.progress(100, text="Report analyzed successfully!")
    time.sleep(0.5)
    progress_bar.empty()

    pattern = r"Name:\s*(.*?)\nActual value:\s*(.*?)\nNormal range:\s*(.*?)\nResult:\s*(.*?)\nTips:\s*(.*?)\n"
    data = re.findall(pattern, explanation_text, re.DOTALL)

    if data:
        df = pd.DataFrame(data, columns=["Test Name", "Actual Value", "Normal Range", "Result", "Tips"])

        def add_result_emoji(result):
            result = result.strip().lower()
            if "high" in result:
                return "🔺 High"
            elif "low" in result:
                return "🔻 Low"
            elif "normal" in result:
                return "✅ Normal"
            elif "borderline" in result:
                return "⚠️ Borderline"
            else:
                return f"❔ {result.capitalize() or 'Not specified'}"

        df["Result"] = df["Result"].apply(add_result_emoji)

        st.success("Explanation generated successfully!")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Report as CSV", csv, "explained_report.csv", "text/csv")

    else:
        st.warning("⚠️ Could not parse structured output. Showing raw explanation below:")
        st.text_area("Raw Output", explanation_text, height=400)
