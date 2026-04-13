from src.data_ingestion import DataIngestion
from src.feature_engineering import engineer_features
from src.preprocessing import build_pipeline
from src.segmentation import CustomerSegmentation
from src.churn_model import ChurnModelTrainer
from src.profit_engine import ProfitOptimizationEngine, plot_profit_curve

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Load Data
ingestion = DataIngestion("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = ingestion.load_data()

# 2. Feature Engineering
df = engineer_features(df)

# 3. Split Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# 4. Preprocessing
pipeline = build_pipeline()
X_processed = pipeline.fit_transform(X)

# 4.1 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Segmentation
seg = CustomerSegmentation()
seg.fit(X_train)
train_clusters = seg.predict(X_train)
test_clusters = seg.predict(X_test)

# 6. Train Models
trainer = ChurnModelTrainer()
trainer.train_global_model(X_train, y_train, X_test, y_test)
trainer.train_cluster_models(
    X_train,
    y_train,
    X_test,
    y_test,
    train_clusters,
    test_clusters,
)

# 7. Predict Probabilities
probs = trainer.predict_probabilities(X_test)

# 8. Profit Optimization
engine = ProfitOptimizationEngine()

eval_df = df.loc[y_test.index, ["customerID", "CLV"]].copy()
eval_df["churn_probability"] = probs
eval_df = eval_df.rename(columns={"CLV": "clv"})

results = engine.optimize_targeting(
    df=eval_df,
    retention_cost=500,
    budget=100000,
)

# 9. Show Top Customers
print("Campaign Summary:")
print(pd.Series(results["campaign_summary"]))

print("\nTop Selected Customers:")
customer_details_df = pd.DataFrame(results["customer_details"])
print(customer_details_df.head(10))

# 10. Generate and save profit curve image
fig, ax, curve_df = plot_profit_curve(customer_details_df)
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)
plot_path = artifacts_dir / "profit_curve.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")