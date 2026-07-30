import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load saved artifacts
# -----------------------------
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")
categorical_options = joblib.load("categorical_options.pkl")
data = pd.read_csv("clean_car_data.csv")

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------- theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0B0E14;
    --panel: #141A24;
    --panel-border: #232C3A;
    --text: #E8ECF1;
    --text-muted: #8B93A3;
    --accent: #F2A93B;
    --accent-2: #37D0C4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(242,169,59,0.05), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(55,208,196,0.05), transparent 40%),
        var(--bg);
}

section[data-testid="stSidebar"] {
    background-color: #0E121B;
    border-right: 1px solid var(--panel-border);
}

h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 52px !important;
    text-align: center;
    color: #FFFFFF !important;
    text-shadow: 0 0 24px rgba(242,169,59,0.25);
    margin-bottom: 4px !important;
}

.sidebar-eyebrow {
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 2px;
    font-size: 12px;
    color: var(--accent);
    font-weight: 600;
    text-transform: uppercase;
    margin-top: 4px;
}
.sidebar-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 14px;
}

.panel-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 12px;
    letter-spacing: 2px;
    color: var(--accent-2);
    text-transform: uppercase;
    font-weight: 600;
    margin: 16px 0 6px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--panel-border); }
.stTabs [data-baseweb="tab"] {
    background-color: var(--panel);
    border: 1px solid var(--panel-border);
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    padding: 10px 22px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 16px;
    color: var(--text-muted);
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background-color: #1B2230 !important;
    border-color: var(--accent) !important;
}

/* Inputs */
div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: var(--panel) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
label { color: var(--text-muted) !important; font-size: 12px !important; font-weight: 500 !important;
        text-transform: uppercase; letter-spacing: 0.5px; }

/* Predict button */
.stButton > button {
    font-family: 'Rajdhani', sans-serif;
    background: linear-gradient(135deg, var(--accent), #d68b1f);
    color: #16110A;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border: none;
    border-radius: 6px;
    padding: 12px 0;
    width: 100%;
    margin-top: 12px;
}
.stButton > button:hover { box-shadow: 0 6px 20px rgba(242,169,59,0.25); color: #16110A; }

/* Sliders */
.stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }
.stSlider [role="slider"] { background-color: var(--accent) !important; border-color: var(--accent) !important; }

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR : ALL INPUTS
# =====================================================

with st.sidebar:
    st.markdown('<div class="sidebar-eyebrow">Cardekho Valuation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🚗 Car Details</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-label">Identity</div>', unsafe_allow_html=True)
    brand = st.selectbox("Brand", categorical_options["brand"])

    # only show models that actually belong to the selected brand
    models_for_brand = sorted(data.loc[data["brand"] == brand, "car_name"].unique().tolist())
    car_name = st.selectbox("Car Model", models_for_brand)

    seller_type = st.selectbox("Seller Type", categorical_options["seller_type"])
    fuel_type = st.selectbox("Fuel Type", categorical_options["fuel_type"])
    transmission_type = st.selectbox("Transmission", categorical_options["transmission_type"])
    seats = st.selectbox("Seats", categorical_options["seats"])

    st.markdown('<div class="panel-label">Condition &amp; Specs</div>', unsafe_allow_html=True)
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (cc)", min_value=0, value=1200, step=50)
    max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=80.0, step=1.0)

    predict_clicked = st.button("Predict Price")

# =====================================================
# MAIN AREA
# =====================================================

st.title("🚗 Used Car Price Predictor")

tab1, tab2 = st.tabs(["🚗 Prediction Result", "📊 Data Visualization"])

# ---------------- TAB 1 : RESULT ----------------
with tab1:

    if predict_clicked:
        km_driven_per_year = round(km_driven / max(vehicle_age, 1), 0)
        power_to_engine_ratio = round(max_power * 1000 / max(engine, 1), 2)

        input_dict = {
            "car_name": car_name,
            "brand": brand,
            "vehicle_age": vehicle_age,
            "km_driven": np.log1p(km_driven),
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "mileage": mileage,
            "engine": np.log1p(engine),
            "max_power": np.log1p(max_power),
            "seats": seats,
            "km_driven_per_year": np.log1p(km_driven_per_year),
            "power_to_engine_ratio": power_to_engine_ratio,
        }

        input_df = pd.DataFrame([input_dict])
        categorical_columns = list(categorical_options.keys())
        encoded = encoder.transform(input_df[categorical_columns])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))

        final_input = pd.concat([input_df.drop(columns=categorical_columns), encoded_df], axis=1)
        final_input = final_input.reindex(columns=feature_columns, fill_value=0)

        log_price = model.predict(final_input)[0]
        price_lakhs = float(np.expm1(log_price))

        result_col = st.columns([1, 2, 1])[1]
        with result_col:
            st.markdown(f"""
            <div style="background:#141A24; border:1px solid #232C3A; border-radius:14px;
                        padding:36px; text-align:center; margin-top:24px;">
                <div style="font-family:'Rajdhani',sans-serif; letter-spacing:2px; color:#8B93A3;
                            font-size:13px; text-transform:uppercase;">Estimated Selling Price</div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:56px; font-weight:700;
                            color:#F2A93B; line-height:1.2;">₹ {price_lakhs:.2f} Lakhs</div>
                <div style="color:#8B93A3; font-size:13px; margin-top:4px;">≈ ₹ {price_lakhs*100000:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        result_col = st.columns([1, 2, 1])[1]
        with result_col:
            st.markdown("""
            <div style="background:#141A24; border:1px solid #232C3A; border-radius:14px;
                        padding:36px; text-align:center; margin-top:24px;">
                <div style="font-family:'Rajdhani',sans-serif; letter-spacing:2px; color:#8B93A3;
                            font-size:13px; text-transform:uppercase;">Awaiting Input</div>
                <div style="color:#8B93A3; font-size:14px; margin-top:8px;">
                    Fill in the car details in the sidebar and click Predict Price.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------- TAB 2 : VISUALIZATION ----------------
with tab2:

    st.header("📊 Exploratory Data Analysis")

    chart = st.selectbox(
        "Choose a Visualization",
        (
            "Selling Price Distribution",
            "Fuel Type Distribution",
            "Transmission Type Distribution",
            "Vehicle Age vs Selling Price",
            "Brand-wise Average Selling Price",
            "Correlation Heatmap",
        ),
    )

    chart_col = st.columns([1, 3, 1])[1]

    if chart == "Selling Price Distribution":
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data["selling_price_in_lakhs"], bins=30, kde=True, ax=ax, color="skyblue")
        ax.set_title("Selling Price Distribution")
        fig.tight_layout()
        with chart_col:
            st.pyplot(fig, use_container_width=True)

    elif chart == "Fuel Type Distribution":
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=data, x="fuel_type", ax=ax)
        ax.set_title("Fuel Type Distribution")
        fig.tight_layout()
        with chart_col:
            st.pyplot(fig, use_container_width=True)

    elif chart == "Transmission Type Distribution":
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=data, x="transmission_type", ax=ax)
        ax.set_title("Transmission Type Distribution")
        fig.tight_layout()
        with chart_col:
            st.pyplot(fig, use_container_width=True)

    elif chart == "Vehicle Age vs Selling Price":
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=data, x="vehicle_age", y="selling_price_in_lakhs", alpha=0.6, ax=ax)
        ax.set_title("Vehicle Age vs Selling Price")
        fig.tight_layout()
        with chart_col:
            st.pyplot(fig, use_container_width=True)

    elif chart == "Brand-wise Average Selling Price":
        brand_price = data.groupby("brand")["selling_price_in_lakhs"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(4, len(brand_price) * 0.28)))
        sns.barplot(x=brand_price.values, y=brand_price.index, ax=ax, orient="h")
        ax.set_xlabel("Average Price (Lakhs)")
        ax.set_ylabel("Brand")
        ax.set_title("Average Selling Price by Brand")
        ax.tick_params(axis='y', labelsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    elif chart == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(7, 6))
        corr = data.select_dtypes(include=np.number).corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        with chart_col:
            st.pyplot(fig, use_container_width=True)
            