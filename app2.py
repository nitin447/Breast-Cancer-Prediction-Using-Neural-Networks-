# app_hybrid_final_v19_3.py (Ultra v19.3 – Final Hybrid: NN + Full Explainability with Advice)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from datetime import datetime
import io

# ---------------- Page Config ----------------
st.set_page_config(page_title="Breast Cancer Predictor (Hybrid)", page_icon="🧬", layout="wide")

# ---------------- Dataset ----------------
breast = load_breast_cancer()
X_full = pd.DataFrame(breast.data, columns=breast.feature_names)
y_full = breast.target

TOP_FEATURES = [
    "mean concavity","worst concavity",
    "mean perimeter","worst perimeter",
    "mean radius","worst radius",
    "mean area","worst area",
    "mean compactness","worst texture"
]

X = X_full[TOP_FEATURES]
X_train, X_test, y_train, y_test = train_test_split(X, y_full, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Neural Network Model
model = MLPClassifier(hidden_layer_sizes=(30,15), activation="relu", solver="adam", max_iter=500, random_state=42)
model.fit(X_train_std, y_train)
y_pred = model.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)

# ---------------- Rule-based Explainability ----------------
feature_means = X.mean()
feature_stds = X.std()

COMPLICATIONS = {
    "mean concavity": "Irregular tumor cell shapes → aggressive spread risk.",
    "worst concavity": "Higher concavity clusters malignant cells.",
    "mean perimeter": "Larger perimeter → tumor spreading irregularly.",
    "worst perimeter": "Wider perimeter → invasive tissue growth.",
    "mean radius": "Larger radius → bigger tumor cells (malignant sign).",
    "worst radius": "Worst radius high → aggressive tumor mass formation.",
    "mean area": "Higher area → increased tumor spread.",
    "worst area": "Worst area large → invasive carcinoma risk.",
    "mean compactness": "High compactness → dense tumor tissue.",
    "worst texture": "Irregular texture → higher malignancy chance."
}

def rule_based_explanation(patient_values, diagnosis):
    table_rows, advice = [], []
    summary_lines = []

    abnormal, borderline, slight, normal = [], [], [], []

    for i, feat in enumerate(TOP_FEATURES):
        val = patient_values[i]
        mean, std = feature_means[feat], feature_stds[feat]
        low, high = mean - std, mean + std
        tol = 0.05 * (high - low)

        if val < low - tol:
            status = f"🔴 Abnormal Low ({val:.2f})"
            abnormal.append(f"{feat} ({val:.2f}) → {COMPLICATIONS.get(feat,'Needs evaluation')}")
            advice.append(f"- {feat} is significantly low → {COMPLICATIONS.get(feat,'Consult doctor')}")
        elif val > high + tol:
            status = f"🔴 Abnormal High ({val:.2f})"
            abnormal.append(f"{feat} ({val:.2f}) → {COMPLICATIONS.get(feat,'Needs urgent check')}")
            advice.append(f"- {feat} is too high → {COMPLICATIONS.get(feat,'Urgent medical consultation required')}")
        elif abs(val - low) <= tol or abs(val - high) <= tol:
            status = f"🟡 Borderline ({val:.2f})"
            borderline.append(f"{feat} ({val:.2f}) → Near threshold")
            advice.append(f"- {feat} is borderline → Monitor every 3–6 months.")
        elif val < low:
            status = f"🟠 Slightly Low ({val:.2f})"
            slight.append(f"{feat} ({val:.2f}) → Mild deviation")
            advice.append(f"- {feat} is slightly low → Lifestyle adjustment recommended.")
        elif val > high:
            status = f"🟠 Slightly High ({val:.2f})"
            slight.append(f"{feat} ({val:.2f}) → Mild deviation")
            advice.append(f"- {feat} is slightly high → Re-test in 3–6 months.")
        else:
            status = f"🟢 Normal ({val:.2f})"
            normal.append(feat)

        table_rows.append([feat, f"{val:.2f}", f"{low:.2f}–{high:.2f}", status])

    # ---- Structured Summary ----
    summary_lines.append("## 📌 Final Summary")

    if abnormal:
        summary_lines.append("### 🔴 Abnormal Features")
        for f in abnormal: summary_lines.append(f"- {f}")

    if borderline:
        summary_lines.append("### 🟡 Borderline Features")
        for f in borderline: summary_lines.append(f"- {f}")

    if slight:
        summary_lines.append("### 🟠 Slight Deviations")
        for f in slight: summary_lines.append(f"- {f}")

    if not abnormal and not borderline and not slight:
        summary_lines.append("### ✅ Normal Parameters")
        summary_lines.append("- All diagnostic parameters are within safe ranges.")
    else:
        summary_lines.append("### ✅ Normal Parameters")
        summary_lines.append("- Except above, all other parameters are within safe range.")

    # Malignant-specific summary
    if "Malignant" in diagnosis:
        summary_lines.append("### ⚠️ Clinical Implications & Next Steps")
        summary_lines.append("- High probability of malignancy progression.")
        summary_lines.append("- Immediate oncologist consultation recommended.")
        summary_lines.append("- Further biopsy, mammogram, and imaging strongly advised.")
        # Default malignant advice
        advice.extend([
            "- Consult an oncologist immediately.",
            "- Schedule advanced imaging (mammogram, MRI, biopsy).",
            "- Avoid delays in treatment planning.",
            "- Maintain proper nutrition & rest until evaluation."
        ])
    else:  # Benign case
        summary_lines.append("### 🟢 Clinical Note & Lifestyle Guidelines")
        summary_lines.append("- No signs of abnormal tumor growth.")
        summary_lines.append("- Regular health checkups every 6–12 months recommended.")
        summary_lines.append("- Maintain balanced diet, hydration, daily exercise.")
        summary_lines.append("- Avoid smoking, alcohol, and manage stress properly.")
        summary_lines.append("- If unusual symptoms appear, consult a doctor immediately.")
        # Default benign advice
        advice.extend([
            "- Continue regular health checkups every 6–12 months.",
            "- Maintain a balanced diet and hydration.",
            "- Engage in daily exercise and stress management.",
            "- Avoid smoking and alcohol.",
            "- Report any unusual breast changes to a doctor immediately."
        ])

    return "\n".join(summary_lines), advice, table_rows

# ---------------- Charts ----------------
def chart_patient_vs_mean(features, patient_values, means):
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(features, patient_values, alpha=0.8, color="#0096c7")
    ax.plot(features, means, color="orange", marker="o", label="Population Mean")
    ax.set_title("Patient vs Population Mean")
    ax.tick_params(axis='x', rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig); buf.seek(0)
    return buf

def chart_pie(probs):
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(probs, labels=["Malignant","Benign"], autopct='%1.1f%%', colors=['red','green'], startangle=90)
    ax.set_title("Prediction Probability")
    fig.savefig(buf, format="png", dpi=120, facecolor="white")
    plt.close(fig); buf.seek(0)
    return buf

# ---------------- PDF Generator ----------------
def generate_pdf(patient_info, diagnosis, risk, confidence, summary, advice, table_rows, chart_bufs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header Title
    c.setFillColorRGB(0.12,0.27,0.54); c.rect(0, height-70, width, 70, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width/2, height-40, "🧬 Breast Cancer Predictor")

    # Patient Info
    y = height - 110; c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Details:"); y -= 20; c.setFont("Helvetica", 11)
    for d in [f"Name: {patient_info['name']}", f"Age: {patient_info['age']}",
              f"Gender: {patient_info['gender']}", f"Patient ID: {patient_info['id']}",
              f"Date: {patient_info['date']}"]:
        c.drawString(60, y, d); y -= 15

    # Diagnosis
    y -= 20
    if "Malignant" in diagnosis: c.setFillColor(colors.HexColor("#f8d7da")); text_color=colors.red
    else: c.setFillColor(colors.HexColor("#d4edda")); text_color=colors.green
    c.rect(45, y-35, width-90, 40, fill=1, stroke=0)
    c.setFillColor(text_color); c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, y-15, f"Diagnosis: {diagnosis} | Risk: {risk} | Confidence: {confidence:.2f}")

    # Parameter Table
    y -= 60; c.setFont("Helvetica-Bold", 12); c.setFillColor(colors.black)
    c.drawString(50, y, "Parameter Details:"); y -= 20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Feature"); c.drawString(180, y, "Patient Value")
    c.drawString(300, y, "Normal Range"); c.drawString(430, y, "Status"); y -= 14
    c.setFont("Helvetica", 9)
    for row in table_rows:
        feat,val,rng,status=row
        c.drawString(50, y, feat); c.drawString(180, y, val)
        c.drawString(300, y, rng); c.drawString(430, y, status)
        y -= 14

    # Summary
    y -= 30; c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Summary & Explanation:"); y -= 20
    c.setFont("Helvetica", 10)
    for line in summary.split("\n"):
        if line.startswith("##") or line.startswith("###"):
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y, line.replace("#","").strip())
            y -= 18
            c.setFont("Helvetica", 10)
        else:
            c.drawString(70, y, line.strip())
            y -= 14

    # Advice
    y -= 20; c.setFont("Helvetica-Bold", 12); c.drawString(50, y, "Professional Advice:"); y -= 16
    c.setFont("Helvetica", 10)
    for tip in advice:
        c.drawString(70, y, tip); y -= 14

    # Visuals on new page
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 80, "📊 Patient Visual Analysis:")
    img1 = ImageReader(chart_bufs[0]); c.drawImage(img1, 40, height - 380, width=250, height=200)
    img2 = ImageReader(chart_bufs[1]); c.drawImage(img2, 320, height - 380, width=250, height=200)

    c.save(); buffer.seek(0); return buffer

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center; color:#1f6feb;'>🧬 Breast Cancer Predictor</h1>", unsafe_allow_html=True)

st.sidebar.title("🔎 Menu")
page = st.sidebar.radio("Navigate", ["🏠 Manual Prediction","📂 Batch Prediction","ℹ️ About & Model"])

if page=="🏠 Manual Prediction":
    st.subheader("✍️ Enter Patient Information")
    col1,col2,col3 = st.columns(3)
    name = col1.text_input("Name","John Doe")
    age = col2.number_input("Age",18,100,30)
    gender = col3.selectbox("Gender",["Male","Female","Other"])

    st.subheader("✍️ Enter Patient Features")
    cols=st.columns(2); user_vals=[]
    for i,f in enumerate(TOP_FEATURES):
        with cols[i%2]: v=st.number_input(f,value=float(X[f].mean()),format="%.4f"); user_vals.append(v)

    if st.button("🔮 Predict (NN + Explainability)"):
        arr=np.asarray(user_vals).reshape(1,-1)
        arr_std=scaler.transform(arr)
        pred=model.predict(arr_std)[0]
        probs=model.predict_proba(arr_std)[0]
        confidence=float(max(probs))

        diagnosis="Malignant (Cancerous)" if pred==0 else "Benign (Non-Cancerous)"
        risk="High" if pred==0 else "Low"

        summary, advice, table_rows=rule_based_explanation(user_vals, diagnosis)

        st.subheader("📊 Result")
        if pred==0: st.error(f"⚠️ {diagnosis}")
        else: st.success(f"✅ {diagnosis}")

        st.metric("Risk Level", risk)
        st.metric("Confidence", f"{confidence*100:.2f}%")

        st.subheader("📑 Parameter Analysis")
        st.table(pd.DataFrame(table_rows, columns=["Feature","Value","Range","Status"]))

        st.subheader("📌 Summary & Explanation")
        st.markdown(summary, unsafe_allow_html=True)

        st.subheader("💡 Professional Advice")
        for tip in advice: st.write(tip)

        buf1=chart_patient_vs_mean(TOP_FEATURES,user_vals,X[TOP_FEATURES].mean().values)
        buf2=chart_pie(probs)
        st.subheader("📈 Visuals")
        st.image(plt.imread(buf1))
        st.image(plt.imread(buf2))

        patient_info={"name":name,"age":age,"gender":gender,
                      "id":f"PID-{np.random.randint(1000,9999)}",
                      "date":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        pdf=generate_pdf(patient_info,diagnosis,risk,confidence,summary,advice,table_rows,[buf1,buf2])
        st.download_button("⬇️ Download PDF Report",data=pdf,
                           file_name=f"breast_report_{patient_info['id']}.pdf",mime="application/pdf")

elif page=="📂 Batch Prediction":
    st.subheader("📂 Batch Prediction Mode")
    uploaded=st.file_uploader("Upload CSV with required 10 features",type="csv")
    if uploaded is not None:
        df=pd.read_csv(uploaded)
        if not all(col in df.columns for col in TOP_FEATURES):
            st.error(f"CSV must include columns: {TOP_FEATURES}")
        else:
            st.success("✅ File loaded")
            df_std=scaler.transform(df[TOP_FEATURES])
            preds=model.predict(df_std)
            df["Prediction"]=["Malignant" if p==0 else "Benign" for p in preds]
            st.dataframe(df)

elif page=="ℹ️ About & Model":
    st.subheader("ℹ️ About this App")
    st.write(f"""
    - **Algorithm:** Neural Network (MLPClassifier) + Rule-based Explainability  
    - **Dataset:** Breast Cancer Wisconsin (sklearn.datasets)  
    - **Features Used:** {TOP_FEATURES}  
    - **Test Accuracy (NN):** {accuracy*100:.2f}%  
    - **Explainability:** Structured summary with abnormal, borderline, normal ranges, clinical notes, and lifestyle advice  
    - **Disclaimer:** Educational demo — not medical diagnosis  
    """)
