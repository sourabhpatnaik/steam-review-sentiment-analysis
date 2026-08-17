import streamlit as st
import joblib
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from preprocessing import cleaning_text


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SteamSentiment AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HTML RENDER FUNCTION (FIXED)
# ============================================================
# Streamlit's markdown renderer treats any line with 4+ leading
# spaces as a code block. Multi-line indented f-strings trigger
# this constantly. Stripping every line individually (instead of
# relying on textwrap.dedent's "common prefix" logic) fixes it
# everywhere, for every call, in one place.

def render_html(html):
    cleaned = "\n".join(line.strip() for line in html.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# CUSTOM CSS (IMPROVED)
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Motiva+Sans:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Motiva Sans', sans-serif;
    }

    /* ==============================
       MAIN APP — Steam's signature
       dark blue, not pure black
       ============================== */

    .stApp {
        background: linear-gradient(180deg, #1b2838 0%, #171a21 45%, #0f141b 100%);
    }

    section.main > div {
        padding-top: 1.5rem;
    }

    /* Steam-style thin top accent bar */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #66c0f4, #4c6b22, #66c0f4);
        background-size: 200% 100%;
        z-index: 999;
        animation: steam-shimmer 6s linear infinite;
    }

    @keyframes steam-shimmer {
        0% { background-position: 0% 0%; }
        100% { background-position: 200% 0%; }
    }

    /* ==============================
       SIDEBAR — Steam's darkest panel
       ============================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171a21 0%, #0f141b 100%);
        border-right: 1px solid #2a3f5a;
    }

    section[data-testid="stSidebar"] * {
        color: #c7d5e0;
    }

    /* ==============================
       TITLES
       ============================== */

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #ffffff;
        letter-spacing: 0.3px;
    }

    .subtitle {
        color: #8f98a0;
        font-size: 16px;
        margin-bottom: 22px;
        line-height: 1.5;
    }

    /* ==============================
       CARDS — Steam capsule style
       ============================== */

    .info-card {
        background: linear-gradient(160deg, #16202d, #0f1621);
        border: 1px solid #2a3f5a;
        border-radius: 4px;
        padding: 22px;
        margin-bottom: 15px;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .info-card:hover {
        border-color: #66c0f4;
        box-shadow: 0 0 16px rgba(102, 192, 244, 0.15);
    }

    .card-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .card-text {
        color: #8f98a0;
        font-size: 14px;
        line-height: 1.7;
    }

    /* ==============================
       POSITIVE / NEGATIVE — Steam
       review-recommendation colors
       (#66c0f4 blue / green tag / red tag)
       ============================== */

    .positive-card {
        background: linear-gradient(160deg, #1c2d1a, #141f13);
        border: 1px solid #a3cf06;
        box-shadow: 0 0 26px rgba(163, 207, 6, 0.18);
        border-radius: 4px;
        padding: 28px;
        text-align: center;
    }

    .negative-card {
        background: linear-gradient(160deg, #331a1a, #221111);
        border: 1px solid #e0542f;
        box-shadow: 0 0 26px rgba(224, 84, 47, 0.18);
        border-radius: 4px;
        padding: 28px;
        text-align: center;
    }

    .neutral-card {
        background: linear-gradient(160deg, #2d2716, #1e1a0f);
        border: 1px solid #cc9a1f;
        box-shadow: 0 0 26px rgba(204, 154, 31, 0.18);
        border-radius: 4px;
        padding: 28px;
        text-align: center;
    }

    .prediction-label {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
    }

    .prediction-confidence {
        font-size: 17px;
        margin-top: 10px;
        color: #c7d5e0;
    }

    /* ==============================
       METRIC CARDS
       ============================== */

    .metric-card {
        background: linear-gradient(160deg, #16202d, #0f1621);
        border: 1px solid #2a3f5a;
        border-radius: 4px;
        padding: 20px 10px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #66c0f4;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #66c0f4;
    }

    .metric-label {
        color: #8f98a0;
        font-size: 12.5px;
        margin-top: 6px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ==============================
       PIPELINE CARDS
       ============================== */

    .pipeline {
        background: linear-gradient(160deg, #16202d, #0f1621);
        border: 1px solid #2a3f5a;
        border-radius: 4px;
        padding: 26px 14px;
        text-align: center;
        min-height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        gap: 6px;
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .pipeline:hover {
        transform: translateY(-3px);
        border-color: #66c0f4;
        box-shadow: 0 4px 20px rgba(102, 192, 244, 0.15);
    }

    .pipeline-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
    }

    .pipeline-description {
        color: #8f98a0;
        font-size: 13px;
    }

    /* ==============================
       BUTTONS — Steam's green "Add
       to Cart" capsule button
       ============================== */

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #a3cf06, #6ba100);
        border: none;
        border-radius: 3px;
        color: #16202d;
        font-weight: 700;
        transition: filter 0.15s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.1);
        color: #16202d;
    }

    div.stButton > button[kind="secondary"] {
        background: #2a3f5a;
        border: 1px solid #3d5877;
        border-radius: 3px;
        color: #c7d5e0;
    }

    /* Tabs / radio nav in sidebar get a Steam blue accent */
    div[role="radiogroup"] label {
        border-radius: 3px;
    }

    /* ==============================
       DIVIDERS
       ============================== */

    hr {
        border-color: #2a3f5a !important;
    }

    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #5c7086;
        font-size: 13px;
        padding-top: 30px;
        padding-bottom: 20px;
        border-top: 1px solid #2a3f5a;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    model = joblib.load("Final_model/xgb_model.pkl")
    w2v_model = Word2Vec.load("Final_model/word2vec.model")
    scaler = joblib.load("Final_model/scaler.pkl")
    return model, w2v_model, scaler


model, w2v_model, scaler = load_models()


# ============================================================
# WORD2VEC DOCUMENT VECTOR
# ============================================================

def get_document_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎮 SteamSentiment AI")
    st.caption("AI-Powered Steam Review Analysis")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🔍 Sentiment Analysis",
            "📁 Batch Analysis (CSV)",
            "📊 Model Information",
            "🏆 Why XGBoost?",
            "📌 Project Overview"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🧠 Final Model
        **XGBoost**

        ### 🔤 Features
        **Word2Vec**

        ### 🎯 Task
        **Binary Sentiment Classification**
        """
    )

    st.markdown("---")
    st.caption("NLP & Machine Learning Project")


# ============================================================
# PAGE 1 — SENTIMENT ANALYSIS
# ============================================================

if page == "🔍 Sentiment Analysis":

    render_html(
        """
        <div class="main-title">🎮 Steam Review Sentiment Analysis</div>
        <div class="subtitle">
            Understand the sentiment behind a Steam game review
            using Natural Language Processing and Machine Learning.
        </div>
        """
    )

    st.divider()

    st.markdown("### 📝 Enter a Game Review")

    SAMPLE_REVIEWS = {
        "— Pick a sample review —": "",
        "😊 Glowing praise": (
            "I've played this game for 50 hours and absolutely loved it. "
            "The gameplay is amazing, the story kept me hooked, and the "
            "combat feels so satisfying. Easily one of the best games I've "
            "played this year!"
        ),
        "😊 Short and positive": "Great game, super fun with friends, highly recommend.",
        "😞 Refund rant": (
            "Unplayable at launch. Constant crashes, terrible optimization, "
            "and the matchmaking is broken. Refunded after two hours, "
            "waste of money."
        ),
        "😞 Mixed but negative": (
            "The art style is nice but the gameplay loop gets repetitive "
            "fast and the monetization is way too aggressive for a full "
            "price game. Can't recommend it right now."
        ),
        "🤔 Sarcastic / tricky": (
            "Oh great, ANOTHER game that crashes every 10 minutes. "
            "Really loving spending 60 dollars on a slideshow."
        ),
    }

    sample_choice = st.selectbox(
        "Sample reviews",
        list(SAMPLE_REVIEWS.keys()),
        label_visibility="collapsed"
    )

    review = st.text_area(
        "Review",
        value=SAMPLE_REVIEWS[sample_choice],
        placeholder=(
            "Example: I've played this game for 50 hours "
            "and absolutely loved it. The gameplay is amazing!"
        ),
        height=170,
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        analyze = st.button("🔍 Analyze", use_container_width=True, type="primary")

    with col2:
        clear = st.button("↻ Clear", use_container_width=True)

    if clear:
        st.rerun()

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    if analyze:

        if not review.strip():
            st.warning("Please enter a review before analyzing.")

        else:
            tokens = cleaning_text(review)

            vector = get_document_vector(tokens)
            vector = vector.reshape(1, -1)

            known_words = sum(1 for word in tokens if word in w2v_model.wv)

            vector_scaled = scaler.transform(vector)

            prediction = model.predict(vector_scaled)[0]
            probabilities = model.predict_proba(vector_scaled)[0]

            negative_probability = probabilities[0] * 100
            positive_probability = probabilities[1] * 100
            confidence = max(negative_probability, positive_probability)

            st.divider()
            st.markdown("### 🎯 Prediction")

            # Confidence tier: below 60% is a toss-up even if the class is "right"
            low_confidence = confidence < 60

            if prediction == 1:
                card_class = "neutral-card" if low_confidence else "positive-card"
                label = "🤔 Leaning Positive" if low_confidence else "😊 Positive Review"
                render_html(
                    f"""
                    <div class="{card_class}">
                        <div class="prediction-label">{label}</div>
                        <div class="prediction-confidence">Confidence: <b>{confidence:.2f}%</b></div>
                    </div>
                    """
                )
                if not low_confidence:
                    st.balloons()
            else:
                card_class = "neutral-card" if low_confidence else "negative-card"
                label = "🤔 Leaning Negative" if low_confidence else "😞 Negative Review"
                render_html(
                    f"""
                    <div class="{card_class}">
                        <div class="prediction-label">{label}</div>
                        <div class="prediction-confidence">Confidence: <b>{confidence:.2f}%</b></div>
                    </div>
                    """
                )

            if low_confidence:
                st.caption("⚠️ The model isn't very sure about this one — confidence is below 60%.")

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            st.markdown("### 📊 Sentiment Probability")

            p1, p2 = st.columns(2)

            with p1:
                st.metric("😞 Negative", f"{negative_probability:.2f}%")
                st.progress(int(negative_probability))

            with p2:
                st.metric("😊 Positive", f"{positive_probability:.2f}%")
                st.progress(int(positive_probability))

            # ------------------------------------------------
            # REVIEW STATISTICS
            # ------------------------------------------------

            st.markdown("### 🔎 Review Analysis")

            c1, c2, c3, c4 = st.columns(4)
            stats = [
                (len(review), "Characters"),
                (len(tokens), "Tokens"),
                (known_words, "Known Words"),
                (w2v_model.vector_size, "Vector Dimensions"),
            ]

            for col, (value, label) in zip([c1, c2, c3, c4], stats):
                with col:
                    render_html(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{value}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """
                    )

            # ------------------------------------------------
            # PROCESSED REVIEW
            # ------------------------------------------------

            st.markdown("### 🧹 Processed Review")

            with st.expander("View preprocessing output"):
                st.write(" ".join(tokens) if tokens else "No valid tokens found.")

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            st.markdown("### ⚙️ Prediction Pipeline")

            pipeline_cols = st.columns(4)
            pipeline_data = [
                ("🧹", "Text Cleaning"),
                ("🔤", "Word2Vec"),
                ("📐", "Scaling"),
                ("🧠", "XGBoost")
            ]

            for col, (icon, name) in zip(pipeline_cols, pipeline_data):
                with col:
                    render_html(
                        f"""
                        <div class="pipeline">
                            <span style="font-size:30px;">{icon}</span>
                            <div class="pipeline-title">{name}</div>
                        </div>
                        """
                    )


# ============================================================
# PAGE 1.5 — BATCH ANALYSIS (CSV)
# ============================================================

elif page == "📁 Batch Analysis (CSV)":

    render_html(
        """
        <div class="main-title">📁 Batch Review Analysis</div>
        <div class="subtitle">
            Upload a CSV of Steam reviews and classify all of them
            in one pass.
        </div>
        """
    )

    st.divider()

    st.markdown("### 📤 Upload CSV")

    st.caption(
        "Your CSV needs a column containing the review text. "
        "You'll pick which column to use after uploading."
    )

    uploaded_file = st.file_uploader(
        "CSV file",
        type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            df = None

        if df is not None:

            if df.empty:
                st.warning("This CSV has no rows.")

            else:
                st.markdown("#### 🧾 Preview")
                st.dataframe(df.head(5), use_container_width=True)

                text_column = st.selectbox(
                    "Which column contains the review text?",
                    df.columns.tolist()
                )

                max_rows = len(df)
                row_limit = st.slider(
                    "How many rows to analyze",
                    min_value=1,
                    max_value=max_rows,
                    value=min(500, max_rows),
                    help="Large CSVs can take a while — start smaller if you're just testing."
                )

                run_batch = st.button(
                    "🚀 Run Batch Analysis",
                    type="primary",
                    use_container_width=False
                )

                if run_batch:

                    work_df = df.head(row_limit).copy()
                    work_df[text_column] = work_df[text_column].fillna("").astype(str)

                    progress_bar = st.progress(0, text="Starting…")

                    predictions = []
                    confidences = []
                    neg_probs = []
                    pos_probs = []

                    total = len(work_df)

                    for i, text in enumerate(work_df[text_column]):

                        tokens = cleaning_text(text)
                        vector = get_document_vector(tokens).reshape(1, -1)
                        vector_scaled = scaler.transform(vector)

                        pred = model.predict(vector_scaled)[0]
                        proba = model.predict_proba(vector_scaled)[0]

                        neg_p = proba[0] * 100
                        pos_p = proba[1] * 100

                        predictions.append("Positive" if pred == 1 else "Negative")
                        confidences.append(max(neg_p, pos_p))
                        neg_probs.append(neg_p)
                        pos_probs.append(pos_p)

                        if i % max(1, total // 100) == 0 or i == total - 1:
                            progress_bar.progress(
                                (i + 1) / total,
                                text=f"Analyzing review {i + 1} of {total}…"
                            )

                    progress_bar.empty()

                    work_df["prediction"] = predictions
                    work_df["confidence_%"] = np.round(confidences, 2)
                    work_df["negative_%"] = np.round(neg_probs, 2)
                    work_df["positive_%"] = np.round(pos_probs, 2)

                    st.session_state["batch_result"] = work_df

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        if "batch_result" in st.session_state:

            result_df = st.session_state["batch_result"]

            st.divider()
            st.markdown("### ✅ Results")

            pos_count = (result_df["prediction"] == "Positive").sum()
            neg_count = (result_df["prediction"] == "Negative").sum()
            total_count = len(result_df)
            avg_conf = result_df["confidence_%"].mean()

            r1, r2, r3, r4 = st.columns(4)
            batch_stats = [
                (total_count, "Reviews Analyzed"),
                (pos_count, "Positive"),
                (neg_count, "Negative"),
                (f"{avg_conf:.1f}%", "Avg Confidence"),
            ]

            for col, (value, label) in zip([r1, r2, r3, r4], batch_stats):
                with col:
                    render_html(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">{value}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                        """
                    )

            st.markdown("#### 📊 Sentiment Split")
            st.bar_chart(
                {"Positive": pos_count, "Negative": neg_count}
            )

            st.markdown("#### 📋 Full Results")
            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name="sentiment_results.csv",
                mime="text/csv",
                type="primary"
            )


# ============================================================
# PAGE 2 — MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    render_html(
        """
        <div class="main-title">📊 Model Information</div>
        <div class="subtitle">
            Understanding the architecture behind the final
            sentiment analysis system.
        </div>
        """
    )

    st.divider()

    st.subheader("🧠 Model Architecture")

    architecture = [
        ("🧹", "Text Preprocessing", "Cleaning & Tokenization"),
        ("🔤", "Word2Vec", "Word Embeddings"),
        ("📐", "Feature Scaling", "MinMaxScaler"),
        ("🧠", "XGBoost", "Binary Classification")
    ]

    cols = st.columns(4)

    for col, (icon, title, description) in zip(cols, architecture):
        with col:
            render_html(
                f"""
                <div class="pipeline">
                    <span style="font-size:35px;">{icon}</span>
                    <div class="pipeline-title">{title}</div>
                    <div class="pipeline-description">{description}</div>
                </div>
                """
            )

    st.divider()

    st.subheader("🔤 Word2Vec Configuration")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.metric("Vector Size", "200")
    with w2:
        st.metric("Window", "10")
    with w3:
        st.metric("Epochs", "50")
    with w4:
        st.metric("Workers", "24")

    st.markdown(
        """
        **Word2Vec** converts words into dense numerical vectors
        that capture relationships between words based on their
        surrounding context.

        Each review is represented by averaging the vectors of
        its known words, producing a **200-dimensional document
        vector**.
        """
    )

    st.divider()

    st.subheader("🧹 Text Preprocessing")

    preprocessing_steps = [
        ("📝", "Raw Review"),
        ("🧹", "Text Cleaning"),
        ("✂️", "Tokenization"),
        ("🚫", "Stopword Removal")
    ]

    cols = st.columns(4)
    for col, (icon, name) in zip(cols, preprocessing_steps):
        with col:
            st.info(f"{icon}  {name}")

    st.divider()

    st.subheader("🏆 Final Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", "90.14%")
    with m2:
        st.metric("Macro F1", "0.82")
    with m3:
        st.metric("False F1", "0.70")
    with m4:
        st.metric("True F1", "0.94")

    st.info(
        "Macro F1 is especially useful because the dataset "
        "contains an imbalanced distribution between the classes."
    )

    st.divider()

    st.subheader("🛠️ Technology Stack")

    technologies = [
        ("🐍", "Python"),
        ("📚", "NLTK"),
        ("🔤", "Gensim / Word2Vec"),
        ("🧠", "Scikit-learn + XGBoost"),
        ("🌐", "Streamlit")
    ]

    tech_cols = st.columns(5)
    for col, (icon, name) in zip(tech_cols, technologies):
        with col:
            render_html(
                f"""
                <div class="pipeline">
                    <span style="font-size:30px;">{icon}</span>
                    <div class="pipeline-title">{name}</div>
                </div>
                """
            )


# ============================================================
# PAGE 3 — WHY XGBOOST
# ============================================================

elif page == "🏆 Why XGBoost?":

    render_html(
        """
        <div class="main-title">🏆 Why XGBoost?</div>
        <div class="subtitle">
            Selecting the final model based on experimental
            performance.
        </div>
        """
    )

    st.divider()

    render_html(
        """
        <div class="positive-card">
            <div class="prediction-label">🥇 XGBoost — Final Model</div>
            <div class="prediction-confidence">Best overall performance among the evaluated models</div>
        </div>
        """
    )

    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", "90.14%")
    with c2:
        st.metric("Macro F1", "0.82")
    with c3:
        st.metric("False Recall", "0.63")
    with c4:
        st.metric("True Recall", "0.96")

    st.divider()

    st.subheader("📊 Model Comparison")

    comparison = {
        "Model": [
            "Logistic Regression", "Gaussian NB", "KNN", "Decision Tree",
            "Random Forest", "XGBoost", "ANN"
        ],
        "Accuracy (%)": [83.79, 55.28, 87.09, 84.22, 86.88, 90.14, 87.59],
        "Macro F1": [0.78, 0.52, 0.78, 0.65, 0.71, 0.82, 0.75]
    }

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.subheader("📈 Accuracy Comparison")

    accuracy_chart = {
        "Logistic Regression": 83.79, "Gaussian NB": 55.28, "KNN": 87.09,
        "Decision Tree": 84.22, "Random Forest": 86.88, "XGBoost": 90.14, "ANN": 87.59
    }
    st.bar_chart(accuracy_chart)

    st.subheader("⚖️ Macro F1 Comparison")

    f1_chart = {
        "Logistic Regression": 0.78, "Gaussian NB": 0.52, "KNN": 0.78,
        "Decision Tree": 0.65, "Random Forest": 0.71, "XGBoost": 0.82, "ANN": 0.75
    }
    st.bar_chart(f1_chart)

    st.divider()

    st.subheader("💡 Why Was XGBoost Selected?")

    reasons = [
        "🥇 Highest accuracy among the evaluated models.",
        "⚖️ Highest Macro F1-score among the evaluated models.",
        "🎯 Better minority-class performance than several other models.",
        "📈 Strong balance between precision and recall.",
        "⚙️ Hyperparameter tuning was performed to optimize the model."
    ]

    for reason in reasons:
        st.markdown(f"- {reason}")

    st.divider()

    st.subheader("⚠️ Class Imbalance Consideration")

    st.markdown(
        """
        The dataset contains approximately **80% True reviews
        and 20% False reviews**.

        Because of this imbalance, accuracy alone was not used
        to determine the final model.

        Precision, recall, F1-score and particularly **Macro F1**
        were also considered during model comparison.
        """
    )

    st.success("Final Selection → XGBoost | 90.14% Accuracy | 0.82 Macro F1")


# ============================================================
# PAGE 4 — PROJECT OVERVIEW
# ============================================================

elif page == "📌 Project Overview":

    render_html(
        """
        <div class="main-title">📌 Project Overview</div>
        <div class="subtitle">
            An end-to-end Natural Language Processing project
            for Steam game reviews.
        </div>
        """
    )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.markdown(
        """
        The objective of this project is to develop a machine
        learning system that automatically classifies Steam game
        reviews into **Positive** and **Negative** sentiment.

        The project covers the complete NLP workflow — from raw
        text preprocessing and feature extraction to model
        training, evaluation, hyperparameter tuning and deployment.
        """
    )

    st.divider()

    st.subheader("📂 Dataset")

    d1, d2, d3 = st.columns(3)
    dataset_cards = [
        ("💬 Data Type", "Steam Game Reviews"),
        ("🎯 Target Variable", "voted_up"),
        ("⚖️ Class Distribution", "Approximately 80% True / 20% False"),
    ]

    for col, (title, text) in zip([d1, d2, d3], dataset_cards):
        with col:
            render_html(
                f"""
                <div class="info-card">
                    <div class="card-title">{title}</div>
                    <div class="card-text">{text}</div>
                </div>
                """
            )

    st.divider()

    st.subheader("🔄 End-to-End NLP Pipeline")

    steps = [
        ("1", "📥", "Raw Reviews"),
        ("2", "🧹", "Text Cleaning"),
        ("3", "✂️", "Tokenization"),
        ("4", "🚫", "Stopword Removal"),
        ("5", "🔤", "Word2Vec"),
        ("6", "📐", "Scaling"),
        ("7", "🧠", "Model Training"),
        ("8", "🎯", "Prediction")
    ]

    for i in range(0, len(steps), 4):
        cols = st.columns(4)
        for col, (number, icon, name) in zip(cols, steps[i:i + 4]):
            with col:
                render_html(
                    f"""
                    <div class="pipeline">
                        <small>STEP {number}</small>
                        <span style="font-size:32px;">{icon}</span>
                        <div class="pipeline-title">{name}</div>
                    </div>
                    """
                )
        if i < 4:
            st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    st.subheader("🤖 Models Evaluated")

    models = [
        "Logistic Regression", "Gaussian Naive Bayes", "K-Nearest Neighbors",
        "Decision Tree", "Random Forest", "XGBoost", "Artificial Neural Network"
    ]

    cols = st.columns(2)
    for i, model_name in enumerate(models):
        with cols[i % 2]:
            st.markdown(f"✓ **{model_name}**")

    st.divider()

    st.subheader("🏆 Final Result")

    r1, r2 = st.columns([1, 2])

    with r1:
        render_html(
            """
            <div class="positive-card">
                <div class="prediction-label">🧠 XGBoost</div>
                <div class="prediction-confidence">Selected Final Model</div>
            </div>
            """
        )

    with r2:
        render_html(
            """
            <div class="info-card">
                <div class="card-title">Final Performance</div>
                <div class="card-text">
                    <b>Accuracy:</b> 90.14%<br>
                    <b>Macro F1:</b> 0.82<br>
                    <b>Feature Representation:</b> Word2Vec<br>
                    <b>Vector Dimension:</b> 200
                </div>
            </div>
            """
        )

    st.divider()

    st.subheader("🛠️ Technology Stack")

    technologies = [
        ("🐍", "Python"),
        ("📚", "NLTK"),
        ("🔤", "Gensim"),
        ("🧠", "XGBoost"),
        ("🌐", "Streamlit")
    ]

    tech_cols = st.columns(5)
    for col, (icon, name) in zip(tech_cols, technologies):
        with col:
            render_html(
                f"""
                <div class="pipeline">
                    <span style="font-size:30px;">{icon}</span>
                    <div class="pipeline-title">{name}</div>
                </div>
                """
            )

    render_html(
        """
        <div class="footer">
            🎮 Steam Review Sentiment Analysis
            <br>
            NLP & Machine Learning Project
        </div>
        """
    )