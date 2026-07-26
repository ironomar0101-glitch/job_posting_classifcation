import pickle as pkl
import pandas as pd
from scipy.sparse import hstack
import streamlit as st


st.set_page_config(
    page_title="Job Fraud Detector AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1120;
        }

        /* Header */
        .hero {
            padding: 1.8rem 2.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #312e81;
            margin-bottom: 1.6rem;
        }
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 4px;
        }
        .sub-title {
            font-size: 1rem;
            color: #94a3b8;
            margin-bottom: 0;
        }

        /* Section cards */
        .section-card {
            background-color: #131c31;
            padding: 1.3rem 1.5rem 0.6rem 1.5rem;
            border-radius: 14px;
            border: 1px solid #1e293b;
            margin-bottom: 1.1rem;
            height: 100%;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #c7d2fe;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Labels */
        label, .stSelectbox label, .stTextInput label, .stTextArea label {
            color: #cbd5e1 !important;
            font-weight: 500;
            font-size: 0.9rem;
        }

        /* Predict button */
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #6366f1, #4338CA);
            color: white;
            font-weight: 700;
            font-size: 1.05rem;
            height: 3.1rem;
            border-radius: 10px;
            border: none;
            transition: all 0.25s ease;
            box-shadow: 0 4px 14px rgba(79,70,229,0.35);
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #4f46e5, #3730a3);
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(79,70,229,0.5);
        }

        /* Result banners */
        .result-fake {
            background: linear-gradient(135deg, #7f1d1d, #3f0d0d);
            border: 1px solid #ef4444;
            padding: 1.5rem 1.8rem;
            border-radius: 16px;
            color: #fecaca;
        }
        .result-real {
            background: linear-gradient(135deg, #064e3b, #022c22);
            border: 1px solid #22c55e;
            padding: 1.5rem 1.8rem;
            border-radius: 16px;
            color: #bbf7d0;
        }
        .result-fake h2, .result-real h2 {
            margin: 0 0 0.4rem 0;
            font-size: 1.5rem;
        }
        .result-fake p, .result-real p {
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.4;
        }

        /* Risk gauge bar */
        .gauge-wrap {
            background-color: #1e293b;
            border-radius: 999px;
            height: 14px;
            width: 100%;
            overflow: hidden;
            margin-top: 0.6rem;
        }
        .gauge-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
        }

        section[data-testid="stSidebar"] {
            background-color: #0b1120;
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] * {
            color: #cbd5e1 !important;
        }

        hr {
            border-color: #1e293b !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    with open('model.pkl', 'rb') as f:
        model = pkl.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pkl.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pkl.load(f)
    with open('columns.pkl', 'rb') as f:
        columns = pkl.load(f)
    return model, scaler, tfidf, columns

model, scaler, tfidf, columns = load()


with st.sidebar:
    st.markdown("## 🛡️ About this tool")
    st.write(
        "An ensemble machine learning model combined with NLP text analysis "
        "to flag job postings that show patterns typical of employment scams."
    )
    st.markdown("---")
    st.markdown("### Usage")
    st.markdown(
        """
        1. Paste the full job description.
        2. Fill in the structural attributes.
        3. Click **Analyze Job Posting**.
        4. Review the fraud risk score and verdict.
        """
    )
    st.markdown("---")
    st.caption(
        "⚠️ This is a statistical estimate, not proof of fraud. "
        "Always verify independently before making decisions."
    )


st.markdown(
    """
    <div class="hero">
        <div class="main-title">🛡️ AI Job Fraud Detection System</div>
        <div class="sub-title">Analyze job postings for fraudulent indicators using Ensemble ML & NLP models.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-card"><div class="section-title">📝 Job Description & Details</div>', unsafe_allow_html=True)
text_input = st.text_area(
    "Job text",
    label_visibility="collapsed",
    placeholder="Paste the full job title, description, requirements, and company info here...",
    height=160,
)
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.subheader("📋 Core Attributes")

required_experience_dic = {
    'Not Applicable': -1,
    'Internship': 0,
    'Entry level': 1,
    'Associate': 2,
    'Mid-Senior level': 3,
    'Director': 4,
    'Executive': 5,
}

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-card"><div class="section-title">🏢 Job Specifications</div>', unsafe_allow_html=True)
    employment_type = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other'])
    industry = st.selectbox('Industry', ['IT', 'Finance', 'Education', 'Marketing', 'Real Estate', 'Other'])
    function_col = st.selectbox('Function', ['Engineering', 'Design', 'Management', 'Sales', 'Other'])
    required_education = st.selectbox(
        'Required Education',
        ["Bachelor's Degree", "High School or equivalent", "Master's Degree",
         "Unspecified", "Associate Degree", "Doctorate", "Other"],
    )
    required_exp_choice = st.selectbox('Required Experience', list(required_experience_dic.keys()))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card"><div class="section-title">⚙️ Risk Indicators (Meta Features)</div>', unsafe_allow_html=True)
    has_company_profile = st.selectbox('Has Company Profile?', [True, False])
    has_company_logo = st.selectbox('Has Company Logo?', [True, False])
    has_questions = st.selectbox('Has Screening Questions?', [True, False])
    has_salary_range = st.selectbox('Has Salary Range Specified?', [True, False])
    telecommuting = st.selectbox('Is Remote / Telecommuting?', [False, True])
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
analyze_clicked = st.button('🔍 Analyze Job Posting')


if analyze_clicked:
    if not text_input.strip():
        st.warning("⚠️ Please provide some text in the Job Description area before predicting.")
    else:
        req_exp_val = required_experience_dic[required_exp_choice]
        logo_val = 1 if has_company_logo else 0
        quest_val = 1 if has_questions else 0
        profile_val = 1 if has_company_profile else 0
        salary_val = 1 if has_salary_range else 0
        tele_val = 1 if telecommuting else 0

        input_dict = {
            'employment_type': employment_type,
            'industry': industry,
            'function': function_col,
            'required_education': required_education,
            'has_company_profile': profile_val,
            'has_company_logo': logo_val,
            'has_questions': quest_val,
            'has_salary_range': salary_val,
            'telecommuting': tele_val,
            'required_experience': req_exp_val,
        }

        input_df = pd.DataFrame([input_dict])
        input_df_encoded = pd.get_dummies(input_df)
        input_df_encoded = input_df_encoded.reindex(columns=columns, fill_value=0)
        input_df_encoded = input_df_encoded.astype(float)

        text_vec = tfidf.transform([text_input])
        combined_features = hstack((input_df_encoded.values, text_vec))
        scaled_features = scaler.transform(combined_features)

        with st.spinner("Analyzing posting..."):
            prediction = model.predict(scaled_features)[0]
            prob = model.predict_proba(scaled_features)[0][1] * 100

        st.write("")
        st.subheader("📊 Analysis Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric(label="Fraud Probability (Risk)", value=f"{prob:.1f}%")
            st.markdown(
                f"""
                <div class="gauge-wrap">
                    <div class="gauge-fill" style="width:{min(prob,100)}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-fake">
                        <h2>🚨 High Risk: Fraudulent Job Posting Detected!</h2>
                        <p>This posting exhibits key patterns commonly found in scam listings.
                        Exercise extreme caution and verify the employer independently.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-real">
                        <h2>✅ Low Risk: Genuine Job Posting</h2>
                        <p>This posting aligns well with legitimate employment listings,
                        based on the patterns the model was trained on.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )