import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from lifelines import KaplanMeierFitter
from lifelines.plotting import plot_lifetimes, add_at_risk_counts
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

st.title("Churn Survival Analysis with User Data Upload and Modeling")

uploaded_file = st.file_uploader("Upload churn dataset CSV", type=["csv", "txt"])

def add_features(df):
    df = df.copy()
    df["day_mins_per_call"] = df["Day Mins"] / (df["Day Calls"] + 1)
    df["eve_mins_per_call"] = df["Eve Mins"] / (df["Eve Calls"] + 1)
    df["charge_per_call"] = df["Night Charge"] / (df["Night Calls"] + 1)
    df["vmail_plan_flag"] = (df["VMail Plan"] == "yes").astype(int)
    return df

def survival_y_cox(dframe: pd.DataFrame) -> np.array:
    y_survival = []
    for idx, row in dframe[["duration", "event"]].iterrows():
        if row["event"]:
            y_survival.append(int(row["duration"]))
        else:
            y_survival.append(-int(row["duration"]))
    return np.array(y_survival)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    # Basic checks and preprocessing like your code
    if ("Churn?" not in df.columns) or ("Account Length" not in df.columns):
        st.error("Dataset must have 'Churn?' and 'Account Length' columns following your original data schema.")
        st.stop()

    df["event"] = np.where(df["Churn?"] == "False.", 0, 1)
    df = df.rename(columns={"Account Length": "duration"})
    df.drop(columns=["Churn?"], inplace=True)
    df = df.dropna().drop_duplicates()

    st.write("Data preview:")
    st.dataframe(df.head())

    st.write(f"Total records: {df.shape[0]}")
    st.write(f"Percent churn rate: {df.event.mean():.4f}")
    st.write("Duration intervals:")
    st.write(df['duration'].describe())

    st.write("Plotting customer attrition lifetimes for first 10 customers:")
    fig, ax = plt.subplots()
    plot_lifetimes(df.head(10)['duration'], df.head(10)['event'], ax=ax)
    ax.set_xlabel("Duration: Account Length (days)")
    ax.set_ylabel("Customer Number")
    ax.set_title("Observed Customer Attrition")
    st.pyplot(fig)

    st.write("Kaplan-Meier survival function:")
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration'], event_observed=df['event'])
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_title('Survival function for telco churn')
    ax.set_xlabel("Duration: Account Length (days)")
    ax.set_ylabel("Churn Risk (Survival Probability)")
    ax.axvline(x=kmf.median_survival_time_, color='r', linestyle='--')
    st.pyplot(fig)

    st.write("Adding feature transformations...")
    df = add_features(df)

    st.write("Splitting dataset into train and test...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)

    numerical_idx = (df.select_dtypes(exclude=['object', 'category'])
                     .drop(['event', 'duration'], axis=1)
                     .columns.tolist())

    categorical_idx = (df.select_dtypes(include=['object', 'category'])
                       .columns.tolist())

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numerical_idx),
            ("categorical", categorical_transformer, categorical_idx)
        ],
        remainder="passthrough"
    )

    st.write("Applying preprocessing pipeline...")
    train_features = preprocessor.fit_transform(df_train.drop(['event', 'duration'], axis=1))
    test_features = preprocessor.transform(df_test.drop(['event', 'duration'], axis=1))

    feature_names = np.hstack((
        np.array(numerical_idx),
        preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_idx)
    )).tolist()

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

    st.write("Making predictions on test set...")
    df_test.loc[:, "preds"] = bst.predict(dm_test, output_margin=True)
    df_test.loc[:, "preds_exp"] = bst.predict(dm_test, output_margin=False)

    st.write("Median predictions by 20 quantiles of duration:")
    fig, ax = plt.subplots()
    df_test.groupby(pd.qcut(df_test['duration'], q=20))['preds_exp'].median().plot(kind="bar", ax=ax)
    ax.set_xlabel("Duration Quantiles")
    ax.set_ylabel("Median Predicted Risk")
    st.pyplot(fig)

    y_train = Surv.from_dataframe("event", "duration", df_train)
    y_test = Surv.from_dataframe("event", "duration", df_test)

    c_index, c_index_se = concordance_index_ipcw(y_train, y_test, df_test['preds'], tau=100)
    st.write(f"C-index: {c_index:.4f}")
    st.write(f"Standard Error: {c_index_se:.4f}")

    st.write("Brier score:")
    times, score = brier_score(y_train, y_test, df_test['preds'], df_test['duration'].max() - 1)
    st.line_chart(score)

    st.write("SHAP summary plot of model feature importance:")
    explainer = shap.TreeExplainer(bst, feature_names=feature_names)
    shap_values = explainer.shap_values(test_features)

    # SHAP summary plot requires matplotlib figure capture
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, pd.DataFrame(test_features, columns=feature_names), show=False)
    st.pyplot(fig)

    idx_sample = st.number_input("Input index to inspect SHAP force plot:", min_value=0, max_value=len(df_test) - 1, value=0)
    st.write(f"Inspecting individual sample index: {idx_sample}")
    shap_fig = shap.force_plot(
        explainer.expected_value,
        shap_values[idx_sample, :],
        pd.DataFrame(test_features, columns=feature_names).iloc[idx_sample, :],
        matplotlib=True
    )
    st.pyplot(shap_fig)

    y_preds = np.exp(df_test.preds)
    y_pred_binary = np.where(y_preds > 0.5, 1, 0)

    st.write(f"Accuracy score: {metrics.accuracy_score(df_test.event, y_pred_binary):.4f}")
    st.write(f"AUC score: {metrics.roc_auc_score(df_test.event, y_pred_binary):.4f}")
    st.text(metrics.classification_report(df_test.event, y_pred_binary))

else:
    st.info("Please upload a churn dataset CSV file to start analysis.")
