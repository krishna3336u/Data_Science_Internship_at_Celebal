import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Page Configuration
st.set_page_config(page_title="Customer Intelligence System", layout="wide", page_icon="📊")

# Load Models and Data
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.pkl')
    kmeans = joblib.load('models/kmeans_model.pkl')
    model = joblib.load('models/final_model.pkl')      # Stacking model from Phase 4
    return scaler, kmeans, model

scaler, kmeans, model = load_models()
df = pd.read_csv('data/Country-data.csv')

# ===================== SIDEBAR =====================
st.sidebar.title("Customer Intelligence System")
st.sidebar.image("https://img.icons8.com/color/512/customer-insights.png", width=150)
page = st.sidebar.radio("Navigate", 
    ["🏠 Home", "📊 EDA", "🎯 Customer Segmentation", "🔮 Predict Segment", "💡 Insights & Recommendations"])

# ===================== HOME =====================
if page == "🏠 Home":
    st.title("Customer Intelligence System")
    st.markdown("### Developed using Classification, Ensemble Learning & Clustering")
    st.write("""
    This system helps international development agencies and NGOs to:
    - Segment countries into meaningful customer groups
    - Predict which segment a country belongs to
    - Provide actionable recommendations for aid and policy making
    """)
    st.success("✅ Project combines K-Means Clustering + XGBoost + Random Forest + Stacking")
    
    st.image("https://source.unsplash.com/random/800x400/?dashboard", use_column_width=True)

# ===================== EDA =====================
elif page == "📊 EDA":
    st.title("Exploratory Data Analysis")
    st.write("Key insights from the dataset used for this Customer Intelligence System.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"Total Countries: **{len(df)}**")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe().T.round(2), use_container_width=True)
    
    st.plotly_chart(px.histogram(df, x='gdpp', color='income', title="GDPP Distribution"), use_container_width=True)

elif page == "🎯 Customer Segmentation":
    st.title("Customer Segmentation (Clustering)")
    st.write("Countries segmented into 3 clusters using K-Means")

    # Expected cluster mapping (temporary)
    cluster_names = {0: "Developed Economies", 1: "Developing Economies", 2: "Underdeveloped Economies"}

    # ---- 1) Build X and predict clusters using saved kmeans+scaler ----
    df_seg = df.copy()
    df_seg.columns = df_seg.columns.astype(str).str.strip().str.lower()

    X = df_seg.drop(columns=["country"], errors="ignore")
    # scaler may require the exact columns used during training
    if hasattr(scaler, "feature_names_in_"):
        X = X.reindex(columns=list(scaler.feature_names_in_), fill_value=0)

    X_scaled = scaler.transform(X)
    df_seg["Cluster"] = kmeans.predict(X_scaled)

    # ---- 2) Create Cluster_Name ----
    df_seg["Cluster_Name"] = df_seg["Cluster"].map(cluster_names)

    tab1, tab2 = st.tabs(["Cluster Profile", "Visualization"])

    with tab1:
        st.dataframe(
            df_seg.groupby("Cluster_Name").mean(numeric_only=True).round(2),
            use_container_width=True
        )

    with tab2:
        fig = px.scatter(
            df_seg, x='income', y='gdpp',
            color='Cluster_Name',
            hover_name='country',
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

# ===================== PREDICT SEGMENT =====================
elif page == "🔮 Predict Segment":
    st.title("🔮 Predict Country Segment")
    st.write("Enter the values below to predict which customer segment the country belongs to.")

    col1, col2 = st.columns(2)

    with col1:
        child_mort = st.slider("Child Mortality", 0, 200, 50)
        exports = st.slider("Exports (% of GDP)", 0, 100, 40)
        health = st.slider("Health Spending", 0, 20, 6)
        imports = st.slider("Imports (% of GDP)", 0, 100, 45)
        income = st.slider("Income", 0, 60000, 12000)

    with col2:
        inflation = st.slider("Inflation", -10, 30, 5)
        life_expec = st.slider("Life Expectancy", 30, 90, 65)
        total_fer = st.slider("Total Fertility", 1.0, 8.0, 3.0)
        gdpp = st.slider("GDPP", 0, 70000, 8000)

    # ---------- Predict button ----------
    if st.button("🔍 Predict Segment", type="primary"):

        input_data = pd.DataFrame(
            [[child_mort, exports, health, imports, income,
            inflation, life_expec, total_fer, gdpp]],
            columns=['child_mort', 'exports', 'health', 'imports', 'income',
                    'inflation', 'life_expec', 'total_fer', 'gdpp']
    )

        # align to scaler feature order
        if hasattr(scaler, "feature_names_in_"):
            input_data = input_data.reindex(columns=list(scaler.feature_names_in_), fill_value=0)

        # scale once
        X_scaled = scaler.transform(input_data)

        # KMeans prediction (this varies correctly in your debug, so use it for final answer)
        kmeans_id = int(kmeans.predict(X_scaled)[0])

        # (Optional) classifier prediction (your classifier collapses, but we can still show it)
        if hasattr(model, "predict"):
            try:
                clf_id = int(model.predict(X_scaled)[0])
            except:
                clf_id = None
        else:
            clf_id = None

        cluster_names = {
            0: "Developed Economies",
            1: "Developing Economies",
            2: "Underdeveloped Economies"
        }

        st.success(f"**Predicted Segment (KMeans):** {cluster_names.get(kmeans_id,'Unknown')} (ID={kmeans_id})")

        if clf_id is not None:
            st.info(f"**Classifier (final_model.pkl) predicted ID:** {clf_id} (may be unreliable if it collapses)")

        # Probabilities (if supported)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            st.write("Predicted probabilities:")
            st.write(pd.Series(proba, index=model.classes_))

    # ---------- Debug button (separate, correct indentation) ----------
    if st.button("Debug: predictions on dataset"):
        df_debug = pd.read_csv("data/Country-data.csv")
        df_debug.columns = df_debug.columns.astype(str).str.strip().str.lower()

        # Build X_debug
        X_debug = df_debug.drop(columns=["country"], errors="ignore")

        # Align to scaler feature order
        if hasattr(scaler, "feature_names_in_"):
            X_debug = X_debug.reindex(columns=list(scaler.feature_names_in_), fill_value=0)

        # Ensure numeric + fill missing
        X_debug = X_debug.apply(pd.to_numeric, errors="coerce")
        if X_debug.isna().sum().sum() > 0:
            X_debug = X_debug.fillna(X_debug.median(numeric_only=True))

        # Scale + predict
        X_debug_scaled = scaler.transform(X_debug)
        preds_debug = model.predict(X_debug_scaled)

        st.write("Classifier prediction counts (value_counts):")
        st.write(pd.Series(preds_debug).value_counts().sort_index())

        kmeans_ids_debug = kmeans.predict(X_debug_scaled)
        st.write("KMeans cluster counts:")
        st.write(pd.Series(kmeans_ids_debug).value_counts().sort_index())
# ===================== INSIGHTS =====================
elif page == "💡 Insights & Recommendations":

    st.header("Insights & Recommendations")

    from pathlib import Path
    import pandas as pd
    import numpy as np
    import joblib

    # Paths (change DATA_PATH if your filename differs)
    DATA_PATH = Path("data/Country-data.csv")
    KMEANS_PATH = Path("models/kmeans_model.pkl")
    SCALER_PATH = Path("models/scaler.pkl")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}")
        st.stop()

    if not KMEANS_PATH.exists() or not SCALER_PATH.exists():
        st.error("Missing models. Need models/kmeans_model.pkl and models/scaler.pkl")
        st.stop()

    # Load models
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ---------- Load + clean data ----------
    df = pd.read_csv(DATA_PATH)
    df = df.copy()

    df.columns = df.columns.astype(str).str.strip().str.lower()
    if "country" not in df.columns:
        st.error("Expected a 'country' column in dataset.")
        st.stop()

    df = df.drop_duplicates()

    numeric_cols = [c for c in df.columns if c != "country"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # ---------- Build X ----------
    X = df.drop(columns=["country"], errors="ignore")

    if hasattr(scaler, "feature_names_in_"):
        X = X.reindex(columns=list(scaler.feature_names_in_), fill_value=np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
    else:
        X = X.select_dtypes(include="number")

    X_scaled = scaler.transform(X)

    # Assign clusters
    df["kmeans_cluster"] = kmeans.predict(X_scaled)

    # Cluster profile
    profile = df.groupby("kmeans_cluster")[numeric_cols].mean().round(2)

    # ---------- Observations logic ----------
    def minmax(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    child_col = "child_mort"
    income_col = "income"
    gdpp_col = "gdpp"
    life_col = "life_expec"

    required = [child_col, income_col, gdpp_col]
    missing = [c for c in required if c not in profile.columns]
    if missing:
        st.error(f"Missing columns in profile: {missing}. Available: {list(profile.columns)}")
        st.stop()

    high_child_cluster = profile[child_col].idxmax()
    high_income_gdpp_cluster = profile[[income_col, gdpp_col]].mean(axis=1).idxmax()

    under_score = profile[child_col] - profile[[income_col, gdpp_col]].mean(axis=1)
    if life_col in profile.columns:
        under_score = under_score - profile[life_col]
    underdeveloped_cluster = under_score.idxmax()

    severity = (
        minmax(profile[child_col]) +
        (1 - minmax(profile[income_col])) +
        (1 - minmax(profile[gdpp_col]))
    )
    if life_col in profile.columns:
        severity = severity + (1 - minmax(profile[life_col]))

    aid_priority_clusters = severity.sort_values(ascending=False).head(2).index.tolist()

    # show 4 observations
    hc_val = float(profile.loc[high_child_cluster, child_col])
    hi_income_val = float(profile.loc[high_income_gdpp_cluster, income_col])
    hi_gdpp_val = float(profile.loc[high_income_gdpp_cluster, gdpp_col])

    ud_child_val = float(profile.loc[underdeveloped_cluster, child_col])
    ud_income_val = float(profile.loc[underdeveloped_cluster, income_col])
    ud_gdpp_val = float(profile.loc[underdeveloped_cluster, gdpp_col])

    aid1, aid2 = aid_priority_clusters[0], aid_priority_clusters[1]

    st.subheader("Cluster Observations")
    st.markdown(f"""
    1) ✅ **High child mortality:** Cluster **{high_child_cluster}** has the highest average **{child_col}** = **{hc_val:.2f}**.

    2) 💰 **High income and GDPP:** Cluster **{high_income_gdpp_cluster}** shows the strongest economic profile:
       - **{income_col}** ≈ **{hi_income_val:.2f}**
       - **{gdpp_col}** ≈ **{hi_gdpp_val:.2f}**

    3) 🏚️ **Underdeveloped cluster:** Cluster **{underdeveloped_cluster}** combines:
       - high **{child_col}** (~{ud_child_val:.2f})
       - lower **{income_col}** (~{ud_income_val:.2f})
       - lower **{gdpp_col}** (~{ud_gdpp_val:.2f})

    4) 🚑 **Aid priority clusters:** Prioritize countries in clusters **{aid1}** and **{aid2}** (highest “severity”).
    """)

    # ---------- Countries to prioritize ----------
    df_severity = df.copy()
    df_severity["severity"] = (
        minmax(df_severity[child_col]) +
        (1 - minmax(df_severity[income_col])) +
        (1 - minmax(df_severity[gdpp_col]))
    )
    if life_col in df_severity.columns:
        df_severity["severity"] = df_severity["severity"] + (1 - minmax(df_severity[life_col]))

    top_countries = (
        df_severity[df_severity["kmeans_cluster"].isin(aid_priority_clusters)]
        .sort_values("severity", ascending=False)
        .head(10)
        [["country", "kmeans_cluster", "severity", child_col, income_col, gdpp_col] +
         ([life_col] if life_col in df_severity.columns else [])]
    )

    st.subheader("Countries to Prioritize for Aid")
    st.caption("Top countries (ranked by severity within the aid-priority clusters).")
    st.dataframe(top_countries.reset_index(drop=True))

st.sidebar.info("Project by Krishna_singh_pce\nWeek-3 Assignment")