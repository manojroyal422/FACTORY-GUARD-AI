# ==========================================
# FactoryGuard AI - Enterprise Edition v2.0
# Advanced Predictive Maintenance System
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import optuna
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_recall_curve
)
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Configuration class for reproducible experiments"""
    random_state: int = 42
    test_size: float = 0.20   # 20% held-out test
    val_size: float  = 0.10   # 10% validation  (70% train total)
    n_trials: int    = 50     # Optuna trials (reduce for faster runs)
    cv_folds: int    = 5
    min_samples_leaf: int = 20

config = Config()
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

# ── Logging ───────────────────────────────────────────────────────────────────
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('factoryguard_training.log', encoding='utf-8'),
        logging.StreamHandler(stream=open(sys.stdout.fileno(),
                                          mode='w', encoding='utf-8',
                                          closefd=False, buffering=1))
    ]
)
logger = logging.getLogger(__name__)

# ── ai4i2020.csv exact column names ──────────────────────────────────────────
# UDI, Product ID, Type,
# Air temperature [K], Process temperature [K],
# Rotational speed [rpm], Torque [Nm], Tool wear [min],
# Machine failure, TWF, HDF, PWF, OSF, RNF
TARGET_RAW   = 'Machine failure'        # exact name in CSV (space, lowercase f)
DROP_COLS    = ['UDI', 'Product ID',    # identifiers
                'TWF', 'HDF', 'PWF',    # sub-failure flags - data leakage
                'OSF', 'RNF']
CAT_COL      = 'Type'
TARGET_CLEAN = 'Machine_failure'        # name after column cleaning step


class FactoryGuardAI:
    """Enterprise-grade predictive maintenance system"""

    def __init__(self, model_dir: str = "FactoryGuardAI"):
        self.model_dir      = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model          = None
        self.scaler         = RobustScaler()
        self.best_threshold = 0.5
        self.feature_names  = None
        self.config         = config

    # ── 1. Load & Validate ────────────────────────────────────────────────────
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate data with comprehensive checks"""
        logger.info("Loading and validating data...")

        df = pd.read_csv(data_path)

        # Log actual columns so mismatches are easy to diagnose
        logger.info(f"CSV columns detected: {df.columns.tolist()}")

        # Validate target exists using the exact CSV name
        assert TARGET_RAW in df.columns, (
            f"Target column '{TARGET_RAW}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )
        assert df[TARGET_RAW].isin([0, 1]).all(), "Invalid target values – expected 0/1"

        # Drop leakage / identifier columns that exist in the file
        df = df.drop([c for c in DROP_COLS if c in df.columns], axis=1)

        # Handle missing values gracefully instead of crashing
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            df = df.fillna(df.median(numeric_only=True))

        logger.info(f"Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        logger.info(f"Class distribution: {dict(df[TARGET_RAW].value_counts())}")
        return df

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    def advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering with domain knowledge"""
        logger.info("Engineering advanced features...")

        df_eng = df.copy()

        # Domain features – use the original spaced column names from the CSV
        df_eng['Torque_Wear']   = df_eng['Torque [Nm]'] * df_eng['Tool wear [min]']
        df_eng['Temp_Diff']     = (df_eng['Process temperature [K]']
                                   - df_eng['Air temperature [K]'])
        df_eng['Speed_Torque']  = (df_eng['Rotational speed [rpm]']
                                   * df_eng['Torque [Nm]'])
        df_eng['Power_kW']      = (df_eng['Rotational speed [rpm]']
                                   * df_eng['Torque [Nm]'] / 9549.3)
        df_eng['Temp_Ratio']    = (df_eng['Process temperature [K]']
                                   / (df_eng['Air temperature [K]'] + 1e-6))
        df_eng['Wear_per_Hour'] = (df_eng['Tool wear [min]']
                                   / (df_eng['Rotational speed [rpm]'] / 60 + 1e-6))

        # Group statistics per machine type
        df_eng['Torque_Std'] = (df_eng.groupby(CAT_COL)['Torque [Nm]']
                                .transform('std').fillna(0))
        df_eng['Temp_Std']   = (df_eng.groupby(CAT_COL)['Process temperature [K]']
                                .transform('std').fillna(0))

        # High-load interaction flag
        df_eng['High_Load'] = (
            (df_eng['Torque [Nm]']     > df_eng['Torque [Nm]'].quantile(0.9)) &
            (df_eng['Tool wear [min]'] > df_eng['Tool wear [min]'].quantile(0.8))
        ).astype(int)

        # Clean all column names: remove brackets, strip whitespace, spaces → _
        df_eng.columns = (
            df_eng.columns
            .str.replace(r'[\[\]]', '', regex=True)
            .str.strip()
            .str.replace(r'\s+', '_', regex=True)
        )
        # After cleaning: 'Machine failure' → 'Machine_failure'  ✓

        # One-hot encode Type (H / L / M)
        df_eng = pd.get_dummies(df_eng, columns=['Type'], drop_first=True)

        # Convert bool columns from get_dummies to int
        bool_cols = df_eng.select_dtypes(include='bool').columns
        df_eng[bool_cols] = df_eng[bool_cols].astype(int)

        n_new = len(df_eng.columns) - len(df.columns)
        logger.info(f"Engineered {n_new} new features - total: {df_eng.shape[1]} columns")
        return df_eng

    # ── 3. Data Splits ────────────────────────────────────────────────────────
    def create_data_splits(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Create stratified train / val / test splits"""
        logger.info("Creating data splits...")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state
        )

        # val_size expressed relative to the remaining (train+val) pool
        relative_val = self.config.val_size / (1.0 - self.config.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=relative_val,
            stratify=y_train_val,
            random_state=self.config.random_state
        )

        logger.info(
            f"Split sizes - Train: {len(X_train):,}  "
            f"Val: {len(X_val):,}  Test: {len(X_test):,}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── 4. Hyperparameter Optimisation ───────────────────────────────────────
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val:   pd.DataFrame, y_val:   pd.Series
    ) -> Dict:
        """Optuna HPO – tunes XGB + LGB + CatBoost jointly"""
        logger.info("Starting hyperparameter optimisation...")

        ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())

        def objective(trial):
            xgb_p = dict(
                objective        = "binary:logistic",
                eval_metric      = "logloss",
                tree_method      = "hist",
                verbosity        = 0,
                random_state     = self.config.random_state,
                scale_pos_weight = ratio,
                max_depth        = trial.suggest_int  ("xgb_max_depth", 3, 8),
                learning_rate    = trial.suggest_float("xgb_lr",     0.01, 0.2),
                n_estimators     = trial.suggest_int  ("xgb_n_est",  200, 600),
                subsample        = trial.suggest_float("xgb_sub",    0.6, 1.0),
                colsample_bytree = trial.suggest_float("xgb_col",    0.6, 1.0),
                gamma            = trial.suggest_float("xgb_gamma",  0.0, 2.0),
            )
            lgb_p = dict(
                objective        = "binary",
                metric           = "binary_logloss",
                verbose          = -1,
                random_state     = self.config.random_state,
                scale_pos_weight = ratio,
                max_depth        = trial.suggest_int  ("lgb_max_depth",   3, 8),
                learning_rate    = trial.suggest_float("lgb_lr",       0.01, 0.2),
                n_estimators     = trial.suggest_int  ("lgb_n_est",    200, 600),
                subsample        = trial.suggest_float("lgb_sub",      0.6, 1.0),
                colsample_bytree = trial.suggest_float("lgb_col",      0.6, 1.0),
                min_child_samples= trial.suggest_int  ("lgb_min_child",  20, 100),
            )
            cat_p = dict(
                loss_function    = "Logloss",
                eval_metric      = "AUC",
                verbose          = False,
                random_seed      = self.config.random_state,
                scale_pos_weight = ratio,
                depth            = trial.suggest_int  ("cat_depth",  3, 8),
                learning_rate    = trial.suggest_float("cat_lr",  0.01, 0.2),
                iterations       = trial.suggest_int  ("cat_iter", 200, 600),
                l2_leaf_reg      = trial.suggest_float("cat_l2",   1.0, 10.0),
            )

            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', XGBClassifier(**xgb_p)),
                    ('lgb', LGBMClassifier(**lgb_p)),
                    ('cat', CatBoostClassifier(**cat_p)),
                ],
                voting='soft'
            )
            ensemble.fit(X_train, y_train)
            preds = ensemble.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)
        logger.info(f"Best validation AUC: {study.best_value:.4f}")
        return study.best_params

    # ── 5. Train Final Ensemble ───────────────────────────────────────────────
    def train_ensemble_model(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        best_params: Dict, ratio: float
    ):
        """Build and fit the production ensemble from best Optuna params"""
        logger.info("Training final ensemble model...")

        p = best_params  # short alias

        xgb = XGBClassifier(
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            tree_method      = "hist",
            verbosity        = 0,
            random_state     = self.config.random_state,
            scale_pos_weight = ratio,
            max_depth        = p.get("xgb_max_depth", 5),
            learning_rate    = p.get("xgb_lr",        0.05),
            n_estimators     = p.get("xgb_n_est",     400),
            subsample        = p.get("xgb_sub",       0.8),
            colsample_bytree = p.get("xgb_col",       0.8),
            gamma            = p.get("xgb_gamma",     0.0),
        )
        lgb = LGBMClassifier(
            objective        = "binary",
            metric           = "binary_logloss",
            verbose          = -1,
            random_state     = self.config.random_state,
            scale_pos_weight = ratio,
            max_depth        = p.get("lgb_max_depth",    5),
            learning_rate    = p.get("lgb_lr",        0.05),
            n_estimators     = p.get("lgb_n_est",     400),
            subsample        = p.get("lgb_sub",       0.8),
            colsample_bytree = p.get("lgb_col",       0.8),
            min_child_samples= p.get("lgb_min_child",  30),
        )
        cat = CatBoostClassifier(
            loss_function    = "Logloss",
            eval_metric      = "AUC",
            verbose          = False,
            random_seed      = self.config.random_state,
            scale_pos_weight = ratio,
            depth            = p.get("cat_depth",    5),
            learning_rate    = p.get("cat_lr",    0.05),
            iterations       = p.get("cat_iter",   400),
            l2_leaf_reg      = p.get("cat_l2",     3.0),
        )

        self.model = VotingClassifier(
            estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat)],
            voting='soft',
            n_jobs=-1
        )

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        self.model.fit(X_train, y_train)
        return self

    # ── 6. Threshold Optimisation ─────────────────────────────────────────────
    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Choose F1-maximising decision threshold on the validation set"""
        logger.info("Optimising decision threshold...")

        val_probs = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

        # precisions/recalls carry one extra element compared to thresholds
        f1s = (2 * precisions[:-1] * recalls[:-1]
               / (precisions[:-1] + recalls[:-1] + 1e-8))
        self.best_threshold = float(thresholds[np.argmax(f1s)])

        logger.info(
            f"Optimal threshold: {self.best_threshold:.3f}  "
            f"(best val F1 = {f1s.max():.4f})"
        )
        return self.best_threshold

    # ── 7. Save Artifacts ─────────────────────────────────────────────────────
    def save_artifacts(self):
        """Persist model, scaler and metadata"""
        logger.info("Saving production artefacts...")

        artifacts = {
            'model'             : self.model,
            'scaler'            : self.scaler,
            'threshold'         : self.best_threshold,
            'feature_names'     : (self.feature_names.tolist()
                                   if self.feature_names is not None else []),
            'config'            : asdict(self.config),
            'training_timestamp': datetime.now().isoformat(),
        }
        for name, obj in artifacts.items():
            joblib.dump(obj, self.model_dir / f"{name}.pkl")

        logger.info(f"Artefacts saved to {self.model_dir.resolve()}")

    # ── 8. Comprehensive Evaluation ───────────────────────────────────────────
    def comprehensive_evaluation(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """Compute and log full test-set metrics"""
        logger.info("Running comprehensive evaluation...")

        test_probs = self.model.predict_proba(X_test)[:, 1]
        y_pred     = (test_probs > self.best_threshold).astype(int)

        metrics = {
            'roc_auc'               : roc_auc_score(y_test, test_probs),
            'f1_score'              : f1_score(y_test, y_pred),
            'classification_report' : classification_report(
                                          y_test, y_pred, output_dict=True),
            'confusion_matrix'      : confusion_matrix(y_test, y_pred).tolist(),
        }

        logger.info(f"Test ROC-AUC : {metrics['roc_auc']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        return metrics

    # ── 9. Interactive Dashboard ──────────────────────────────────────────────
    def generate_interactive_reports(
        self,
        X_test:     pd.DataFrame,
        y_test:     pd.Series,
        test_probs: np.ndarray,
        metrics:    Dict
    ):
        """Build and save an interactive Plotly executive dashboard"""
        logger.info("Generating interactive dashboard...")

        results = X_test.copy()
        results['Failure_Probability'] = test_probs
        results['Predicted_Failure']   = (test_probs > self.best_threshold).astype(int)
        results['Risk_Level'] = pd.cut(
            test_probs,
            bins=[0.0, 0.4, 0.7, 1.0],
            labels=['LOW RISK', 'MEDIUM RISK', 'HIGH RISK'],
            include_lowest=True
        )
        results.to_csv(self.model_dir / 'risk_assessment.csv', index=False)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'ROC Curve', 'Confusion Matrix',
                'Risk Distribution', 'Top 10 Highest-Risk Machines'
            ),
            specs=[[{"type": "xy"},  {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr,
                       name=f"ROC (AUC={metrics['roc_auc']:.3f})",
                       line=dict(color='crimson', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random baseline',
                       line=dict(dash='dash', color='grey')),
            row=1, col=1
        )

        # Confusion matrix heatmap
        cm = metrics['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted: No Failure', 'Predicted: Failure'],
                y=['Actual: No Failure',    'Actual: Failure'],
                colorscale='Blues',
                text=[[str(cm[0][0]), str(cm[0][1])],
                      [str(cm[1][0]), str(cm[1][1])]],
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=False
            ),
            row=1, col=2
        )

        # Risk distribution
        risk_dist = results['Risk_Level'].value_counts().reindex(
            ['LOW RISK', 'MEDIUM RISK', 'HIGH RISK'], fill_value=0
        )
        fig.add_trace(
            go.Bar(
                x=risk_dist.index.tolist(),
                y=risk_dist.values,
                marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
                showlegend=False
            ),
            row=2, col=1
        )

        # Top-10 highest-risk machines
        top10 = results.nlargest(10, 'Failure_Probability')
        fig.add_trace(
            go.Bar(
                x=[f"Machine {i}" for i in range(len(top10))],
                y=top10['Failure_Probability'].values,
                marker_color='#e74c3c',
                showlegend=False
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=850,
            title_text="FactoryGuard AI – Executive Dashboard",
            title_font_size=20
        )
        out_path = self.model_dir / 'executive_dashboard.html'
        fig.write_html(str(out_path))
        logger.info(f"Dashboard saved to {out_path.resolve()}")

    # ── 10. SHAP Explainability ───────────────────────────────────────────────
    def shap_analysis(self, X_test: pd.DataFrame):
        """SHAP feature-importance plots via the XGBoost sub-model"""
        logger.info("Generating SHAP explanations...")

        # VotingClassifier.estimators_ holds fitted estimators directly (not tuples)
        xgb_model   = self.model.estimators_[0]
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)         # (n_samples, n_features)

        joblib.dump(shap_values, self.model_dir / 'shap_values.pkl')
        joblib.dump(explainer,   self.model_dir / 'shap_explainer.pkl')

        # Global summary plot
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(self.model_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Waterfall for the single highest-risk sample
        risk_scores = self.model.predict_proba(X_test)[:, 1]
        top_idx     = int(np.argmax(risk_scores))   # positional index (not df label)

        base = explainer.expected_value
        # expected_value may be list/array for some XGB versions – flatten to scalar
        if isinstance(base, (list, np.ndarray)):
            base = float(np.array(base).ravel()[-1])

        shap.waterfall_plot(
            shap.Explanation(
                values       = shap_values[top_idx],
                base_values  = base,
                data         = X_test.iloc[top_idx].values,
                feature_names= list(X_test.columns)
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(self.model_dir / 'top_risk_explanation.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("SHAP plots saved.")

    # ── Full Pipeline ─────────────────────────────────────────────────────────
    def run_full_pipeline(self, data_path: str) -> Dict:
        """Execute the complete end-to-end production pipeline"""
        logger.info("[ START ] FactoryGuard AI Enterprise Pipeline...")

        # 1. Load & validate
        df = self.load_and_validate_data(data_path)

        # 2. Feature engineering
        #    After cleaning, 'Machine failure' → 'Machine_failure'
        df_eng = self.advanced_feature_engineering(df)

        X = df_eng.drop(TARGET_CLEAN, axis=1)
        y = df_eng[TARGET_CLEAN]
        self.feature_names = X.columns   # store before scaling

        # 3. Scale
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # 4. Split
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.create_data_splits(X_scaled, y)

        # 5. HPO
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)

        # 6. Train
        ratio = float((y_train == 0).sum()) / float((y_train == 1).sum())
        self.train_ensemble_model(X_train, y_train, best_params, ratio)

        # 7. Threshold
        self.optimize_threshold(X_val, y_val)

        # 8. Evaluate
        metrics = self.comprehensive_evaluation(X_test, y_test)

        # 9. Reports & SHAP
        test_probs = self.model.predict_proba(X_test)[:, 1]
        self.generate_interactive_reports(X_test, y_test, test_probs, metrics)
        self.shap_analysis(X_test)

        # 10. Save
        self.save_artifacts()

        logger.info("[ DONE ] Pipeline complete.")
        return metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    system  = FactoryGuardAI(model_dir="FactoryGuardAI")
    metrics = system.run_full_pipeline(r"C:\Users\manoj\Downloads\ai4i2020.csv")

    print("\n" + "=" * 45)
    print("  FactoryGuard AI – Final Test Metrics")
    print("=" * 45)
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")
    print("=" * 45)
