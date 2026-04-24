import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
N_SAMPLES = 200
FEATURES = [
    "setup_min",
    "downtime_min",
    "batch_size",
    "machine_age_yr",
    "operator_exp_yr",
]
TARGET = "cycle_time_min"


def apply_presentation_theme():
    """Inject a presentation-style visual system for a stronger classroom experience."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --bg-primary: #f2f5f3;
            --bg-card: #ffffff;
            --ink-main: #102a25;
            --ink-muted: #4b635c;
            --accent-a: #0f766e;
            --accent-b: #84cc16;
            --line-soft: #d8e4df;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 20%, #dff4ea 0%, transparent 35%),
                radial-gradient(circle at 88% 10%, #e5f3d3 0%, transparent 30%),
                linear-gradient(180deg, #f8fbf9 0%, var(--bg-primary) 100%);
            color: var(--ink-main);
        }

        h1, h2, h3, .stTabs [data-baseweb="tab"] {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0.2px;
        }

        p, li, .stCaption, .stMarkdown, .stMetric {
            font-family: 'DM Sans', sans-serif;
        }

        .hero-wrap {
            background: linear-gradient(125deg, #0f766e 0%, #1b4332 55%, #84cc16 100%);
            border-radius: 18px;
            color: white;
            padding: 1.1rem 1.3rem 1.2rem 1.3rem;
            box-shadow: 0 14px 34px rgba(16, 42, 37, 0.25);
            animation: riseIn 0.75s ease-out;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
        }

        .hero-sub {
            margin-top: 0.35rem;
            font-size: 1rem;
            line-height: 1.4;
            opacity: 0.95;
        }

        div[data-testid="stMetric"] {
            background: var(--bg-card);
            border: 1px solid var(--line-soft);
            border-radius: 14px;
            padding: 0.55rem 0.8rem;
            box-shadow: 0 8px 20px rgba(12, 42, 36, 0.08);
            animation: riseIn 0.55s ease-out;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: #eaf1ee;
            border-radius: 12px;
            padding: 0.45rem 0.9rem;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(140deg, #b8e1d4 0%, #def2c9 100%);
            color: #0b3a33;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e7efe8 0%, #dfe9e3 100%);
            border-right: 1px solid #d0ddd7;
        }

        section[data-testid="stSidebar"] h2 {
            font-size: 1.05rem;
        }

        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header():
    """Render a high-impact header suitable for classroom presentation."""
    st.markdown(
        """
        <div class="hero-wrap">
            <p class="hero-title">IE Predictive Analytics Dashboard</p>
            <p class="hero-sub">
                Demonstrasi Regresi Linear untuk prediksi cycle time produksi.<br>
                Studi kasus pembelajaran: Teknik Industri Semester 4.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_synthetic_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate realistic production data for cycle-time prediction."""
    rng = np.random.default_rng(seed)

    setup_min = rng.uniform(8, 55, n_samples)
    downtime_min = rng.uniform(0, 35, n_samples)
    batch_size = rng.integers(20, 260, n_samples)
    machine_age_yr = rng.uniform(1, 16, n_samples)
    operator_exp_yr = rng.uniform(0.5, 14, n_samples)

    # Synthetic process equation + stochastic noise.
    noise = rng.normal(0, 3.8, n_samples)
    cycle_time_min = (
        22
        + 0.65 * setup_min
        + 0.52 * downtime_min
        + 0.055 * batch_size
        + 0.82 * machine_age_yr
        - 0.72 * operator_exp_yr
        + noise
    )

    data = pd.DataFrame(
        {
            "setup_min": setup_min,
            "downtime_min": downtime_min,
            "batch_size": batch_size,
            "machine_age_yr": machine_age_yr,
            "operator_exp_yr": operator_exp_yr,
            "cycle_time_min": cycle_time_min,
        }
    )
    return data.round(2)


def train_model(data: pd.DataFrame):
    """Run workflow: split -> train -> predict -> evaluate -> interpret artifacts."""
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    coefficients = pd.DataFrame(
        {
            "feature": FEATURES,
            "beta": model.coef_,
            "interpretation": [
                (
                    f"Jika fitur {f} naik 1 unit, maka cycle time berubah "
                    f"sebesar {coef:.3f} menit, variabel lain dianggap konstan."
                )
                for f, coef in zip(FEATURES, model.coef_)
            ],
        }
    )

    residual_df = pd.DataFrame(
        {
            "actual_cycle_time": y_test.values,
            "predicted_cycle_time": y_pred,
            "residual": y_test.values - y_pred,
        }
    )

    artifacts = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "coefficients": coefficients,
        "residual_df": residual_df,
    }
    return artifacts


def get_sidebar_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Create live feature sliders and return one-row dataframe for prediction."""
    st.sidebar.markdown("### Control Panel Produksi")
    st.sidebar.caption("Ubah nilai fitur untuk melihat prediksi cycle time secara live.")

    values = {}
    for feature in FEATURES:
        min_v = float(data[feature].min())
        max_v = float(data[feature].max())
        mean_v = float(data[feature].mean())

        step = 1.0 if feature == "batch_size" else 0.1
        values[feature] = st.sidebar.slider(
            label=feature,
            min_value=float(np.floor(min_v)),
            max_value=float(np.ceil(max_v)),
            value=float(round(mean_v, 1)),
            step=step,
        )

    values["batch_size"] = int(values["batch_size"])
    return pd.DataFrame([values])


def plot_residuals(residual_df: pd.DataFrame):
    """Plot residual diagnostics to assess linearity and error spread."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#f8fbf9")
    ax.set_facecolor("#ffffff")

    sns.scatterplot(
        data=residual_df,
        x="predicted_cycle_time",
        y="residual",
        color="#0f766e",
        s=65,
        alpha=0.85,
        ax=ax,
    )

    ax.axhline(0, color="#dc2626", linestyle="--", linewidth=1.8)
    ax.set_title("Residual Plot (Aktual - Prediksi)", fontsize=13)
    ax.set_xlabel("Prediksi Cycle Time (menit)")
    ax.set_ylabel("Residual (menit)")
    return fig


def build_app():
    st.set_page_config(page_title="IE Predictive Analytics Dashboard", layout="wide")

    apply_presentation_theme()
    render_hero_header()

    data = generate_synthetic_data()
    artifacts = train_model(data)

    user_input = get_sidebar_inputs(data)
    live_pred = artifacts["model"].predict(user_input)[0]

    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1:
        st.metric("Prediksi Cycle Time (Live)", f"{live_pred:.2f} menit")
    with c2:
        st.metric(
            "Data Latih",
            f"{len(artifacts['X_train'])} baris",
            help="Proporsi split 80% untuk training.",
        )
    with c3:
        st.metric(
            "Data Uji",
            f"{len(artifacts['X_test'])} baris",
            help="Proporsi split 20% untuk evaluasi objektif.",
        )

    st.info("Workflow: Data Split (80/20) -> Train -> Predict -> Evaluate -> Interpret")

    with st.expander("Lihat Snapshot Data Sintetis", expanded=False):
        st.dataframe(data.head(10), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(
        ["Performance Metrics", "Diagnostic Plot", "Insight & Coefficients"]
    )

    with tab1:
        m1, m2, m3 = st.columns(3)
        metrics = artifacts["metrics"]

        m1.metric(
            "MAE",
            f"{metrics['mae']:.2f}",
            help=f"Model meleset rata-rata {metrics['mae']:.2f} menit.",
        )
        m2.metric(
            "RMSE",
            f"{metrics['rmse']:.2f}",
            help="RMSE menghukum error besar, penting untuk risiko keterlambatan deadline.",
        )
        m3.metric(
            "R^2",
            f"{metrics['r2']:.3f}",
            help="Proporsi variasi cycle time yang dijelaskan model.",
        )

        st.markdown(
            "- **MAE** mudah dibaca dalam satuan menit.\n"
            "- **RMSE** lebih sensitif terhadap error besar.\n"
            "- **R^2** mengukur goodness-of-fit secara umum."
        )

    with tab2:
        fig = plot_residuals(artifacts["residual_df"])
        st.pyplot(fig)
        st.caption(
            "Catatan: Jika residual membentuk pola melengkung, itu indikasi hubungan non-linear "
            "dan model linear bisa kurang tepat."
        )

    with tab3:
        coef_df = artifacts["coefficients"].copy()
        coef_df["beta"] = coef_df["beta"].round(4)
        st.dataframe(coef_df, use_container_width=True)

        st.subheader("Narasi Otomatis Koefisien")
        for text in coef_df["interpretation"].tolist():
            st.write(f"- {text}")

    st.divider()
    st.subheader("Anti-Miskonsepsi")
    st.warning("R^2 tinggi tidak selalu menjamin model sehat; bisa terjadi data leakage.")
    st.error("Koefisien regresi bukan bukti kausalitas mutlak.")
    st.warning(
        "Waspadai data leakage: fitur dari masa depan atau informasi target yang bocor "
        "akan membuat evaluasi bias."
    )
    st.error(
        "Hindari ekstrapolasi di luar jangkauan data historis karena prediksi bisa menyesatkan."
    )


if __name__ == "__main__":
    build_app()
