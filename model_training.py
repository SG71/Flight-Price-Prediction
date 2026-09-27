import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

NUMERIC_FEATURES = ["duration", "days_left"]
NOMINAL_FEATURES = ["airline", "source", "departure", "arrival", "destination", "class"]
ORDINAL_FEATURES = ["stops"]


def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["zero", "one", "two_or_more"]])),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("nominal", nominal_transformer, NOMINAL_FEATURES),
        ("ordinal", ordinal_transformer, ORDINAL_FEATURES),
    ])


def clean_data(df):
    df = df.drop(columns=["flight"], errors="ignore")
    for col in ["duration", "price"]:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def train_and_save(train_csv="data/train.csv", output_path="model/model.pkl"):
    train_df = pd.read_csv(train_csv)
    train_df = clean_data(train_df)

    X = train_df.drop(columns=["price", "id"], errors="ignore")
    y = train_df["price"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    rf = Pipeline([("preprocessor", preprocessor),
                   ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])
    xgb = Pipeline([("preprocessor", preprocessor),
                     ("model", XGBRegressor(n_estimators=500, learning_rate=0.05,
                                             max_depth=8, subsample=0.8,
                                             colsample_bytree=0.8, random_state=42, n_jobs=-1))])
    lgbm = Pipeline([("preprocessor", preprocessor),
                      ("model", LGBMRegressor(n_estimators=400, learning_rate=0.1,
                                               num_leaves=31, random_state=42,
                                               n_jobs=-1, verbosity=-1))])

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    voting_model = VotingRegressor(
        estimators=[("RF", rf), ("XGB", xgb), ("LGB", lgbm)],
        weights=[0.40, 0.40, 0.20],
        n_jobs=-1,
    )
    voting_model.fit(X_train, y_train)

    val_pred = voting_model.predict(X_val)
    print(f"Validation R2   : {r2_score(y_val, val_pred):.4f}")
    print(f"Validation RMSE : {np.sqrt(mean_squared_error(y_val, val_pred)):.4f}")

    with open(output_path, "wb") as f:
        pickle.dump(voting_model, f)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    train_and_save()
