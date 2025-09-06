import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from lifelines import KaplanMeierFitter
from lifelines.plotting import plot_lifetimes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, brier_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = [7.2, 4.8]
pd.set_option("display.float_format", lambda x: "%.4f" % x)
sns.set_style('darkgrid')
SEED = 123

st.set_page_config(page_title="Churn Survival Analysis", layout="wide")

st.title("🚀 Churn Survival Analysis with User Data Upload & Modeling")

st.markdown(
    """
    ### Instructions:
    - Upload a CSV file with at least the following columns:
        - **Churn?**: churn indicator (e.g., 'False.' for no churn)
        - **Account Length**: numeric duration
        - Other columns like Day Mins, Day Calls, Eve Mins, Eve Calls, Night Charge, Night Calls, VMail Plan will be used for feature engineering.
    - The app will train a survival model and display survival analyses and interactive SHAP explainability plots.
    """
)

uploaded_file = st.file_uploader("Upload churn dataset CSV", type=["csv", "txt"])

def survival_y_cox(dframe: pd.DataFrame) -> np.array:
    y_survival = []
    for _, row in dframe[["duration", "event"]].iterrows():
        y_survival.append(int(row["duration"]) if row["event"] else -int(row["duration"]))
    return np.array(y_survival)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    required_cols = ['Churn?', 'Account Length', 'Day Mins', 'Day Calls', 'Eve Mins', 'Eve Calls', 'Night Charge', 'Night Calls', 'VMail Plan']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Dataset is missing required columns: {missing_cols}")
        st.stop()

    df["event"] = np.where(df["Churn?"] == "False.", 0, 1)
    df = df.rename(columns={"Account Length": "duration"})
    df.drop(columns=["Churn?"], inplace=True)
    df = df.dropna().drop_duplicates()

    # Sidebar info
    with st.sidebar:
        st.header("Dataset Overview")
        st.write(f"🔹 Total records: {df.shape[0]}")
        st.write(f"🔹 Percent churn rate: {df.event.mean():.4f}")
        st.write("🔹 Duration statistics:")
        st.write(df['duration'].describe())

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Customer attrition lifetimes plot
    st.subheader("Observed Customer Attrition (First 10 Customers)")
    fig, ax = plt.subplots()
    plot_lifetimes(df.head(10)['duration'], df.head(10)['event'], ax=ax)
    ax.set_xlabel("Duration: Account Length (days)")
    ax.set_ylabel("Customer Number")
    ax.set_title("Observed Customer Attrition")
    st.pyplot(fig)

    # Kaplan-Meier survival curve with median line and legend
    st.subheader("Kaplan-Meier Survival Function")
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration'], event_observed=df['event'])
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_title('Survival function for churn')
    ax.set_xlabel("Duration: Account Length (days)")
    ax.set_ylabel("Survival Probability")
    ax.axvline(x=kmf.median_survival_time_, color='r', linestyle='--', label='Median Survival Time')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    st.pyplot(fig)

    # Feature engineering
    st.subheader("Feature Engineering & Model Training")

    st.write("Adding new features based on call usage ratios and voicemail plan...")
    df["day_mins_per_call"] = df["Day Mins"] / (df["Day Calls"] + 1)
    df["eve_mins_per_call"] = df["Eve Mins"] / (df["Eve Calls"] + 1)
    df["charge_per_call"] = df["Night Charge"] / (df["Night Calls"] + 1)
    df["vmail_plan_flag"] = (df["VMail Plan"] == "yes").astype(int)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

    numerical_cols = df.select_dtypes(exclude=['object', 'category']).drop(['event', 'duration'], axis=1).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="passthrough")

    st.write("Applying preprocessing...")
    train_features = preprocessor.fit_transform(df_train.drop(['event', 'duration'], axis=1))
    test_features = preprocessor.transform(df_test.drop(['event', 'duration'], axis=1))

    feature_names = np.hstack([
        numerical_cols,
        preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    ]).tolist()

    dm_train = xgb.DMatrix(train_features, label=survival_y_cox(df_train), feature_names=feature_names)
    dm_test = xgb.DMatrix(test_features, label=survival_y_cox(df_test), feature_names=feature_names)

    params = {
        "eta": 0.1,
        "max_depth": 3,
        "objective": "survival:cox",
        "tree_method": "hist",
        "subsample": 0.8,
        "seed": SEED
    }

    st.write("Training XGBoost Cox proportional hazards model...")
    bst = xgb.train(
        params,
        dm_train,
        num_boost_round=1000,
        evals=[(dm_train, "train"), (dm_test, "test")],
        verbose_eval=10,
        early_stopping_rounds=10
    )

    st.subheader("Model Predictions and Evaluation")

    df_test["preds"] = bst.predict(dm_test, output_margin=True)
    df_test["preds_exp"] = bst.predict(dm_test, output_margin=False)

    fig, ax = plt.subplots()
    df_test.groupby(pd.qcut(df_test['duration'], q=20))['preds_exp'].median().plot(kind="bar", ax=ax)
    ax.set_xlabel("Duration Quantiles")
    ax.set_ylabel("Median Predicted Risk")
    st.pyplot(fig)

    y_train = Surv.from_dataframe("event", "duration", df_train)
    y_test = Surv.from_dataframe("event", "duration", df_test)

    c_index = concordance_index_ipcw(y_train, y_test, df_test['preds'], tau=100)
    times, bscores = brier_score(y_train, y_test, df_test['preds'], df_test['duration'].max() - 1)


    st.subheader("Model Explainability with SHAP")

    explainer = shap.TreeExplainer(bst, feature_names=feature_names)
    shap_values = explainer.shap_values(test_features)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, pd.DataFrame(test_features, columns=feature_names), show=False)
    st.pyplot(fig)

    sample_idx = st.number_input("Select individual index for SHAP force plot:", 0, len(df_test) - 1, 0)
    st.write(f"SHAP force plot for individual index: {sample_idx}")

    shap_fig = shap.force_plot(
        explainer.expected_value,
        shap_values[sample_idx, :],
        pd.DataFrame(test_features, columns=feature_names).iloc[sample_idx, :],
        matplotlib=True
    )
    st.pyplot(shap_fig)

    y_preds = np.exp(df_test["preds"])
    y_pred_binary = (y_preds > 0.5).astype(int)

else:
    st.info("Please upload a churn dataset CSV file to start analysis.")
