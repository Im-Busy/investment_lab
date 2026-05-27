"""
Machine Learning Module for Trading

Provides ML components for:
- Feature engineering from OHLCV data
- Regime classification using supervised learning
- Signal quality scoring using trained classifiers
- Unified pipeline orchestrating all ML components
- Experiment logging, purged CV, feature store, metrics, registry, backtest bridge

Modern ML Models (2025):
- CatBoost: Gradient boosting with ordered boosting for time series
- Chronos-2: Amazon's zero-shot time series foundation model
- FinCast: Financial foundation model (pending release)
- xLSTM: Extended LSTM with exponential gating (NeurIPS 2024)

Part A Infrastructure:
    from src.ml.experiment_logger import ExperimentLogger
    from src.ml.purged_cv import PurgedKFold
    from src.ml.feature_store import FeatureStore
    from src.ml.metrics import compute_ic, compute_rank_ic, ic_summary
    from src.ml.registry import ModelRegistry
    from src.ml.backtest_bridge import BacktestBridge
"""

from src.ml.experiment_logger import ExperimentLogger
from src.ml.purged_cv import PurgedKFold
from src.ml.feature_store import FeatureStore
from src.ml.metrics import (
    compute_ic,
    compute_rank_ic,
    compute_information_ratio,
    ic_summary,
    filter_features_by_ic,
)
from src.ml.registry import ModelRegistry
from src.ml.backtest_bridge import BacktestBridge, MLBacktestResult

from src.ml.features import FeatureEngineer
from src.ml.regime_model import RegimeClassifier, MLRegimeState
from src.ml.regime_base import RegimeDetectorBase, RegimeSummary, RegimeType
from src.ml.hmm_regime import HMMRegimeDetector
from src.ml.pca_kmeans_regime import PCAKMeansRegimeDetector
from src.ml.r2_rd_regime import R2RDRegimeDetector
from src.ml.regimefolio import RegimeFolioDetector
from src.ml.macro_regime import MacroRegimeDetector
from src.ml.path_signature_regime import PathSignatureRegimeDetector
from src.ml.signal_scorer import SignalScorer, ScoredSignal
from src.ml.pipeline import MLPipeline, PipelineResult
from src.ml.models import (
    CatBoostForecaster,
    ChronosForecaster,
    FinCastForecaster,
    xLSTMForecaster,
)

# Additional components (used by scripts and tests)
from src.ml.model_selector import ModelSelector, ModelConfig, ModelResult, Recommendation
from src.ml.pattern_classifier import PatternClassifier
from src.ml.feature_engineering import FeatureExtractor, FeatureConfig
from src.ml.feature_selector import FeatureSelector, SelectionResult
from src.ml.ensemble_regime import EnsembleRegimeDetector, EnsembleStrategy
from src.ml.cross_asset_features import (
    CrossAssetFeatures,
    prepare_cross_asset_data,
    CrossAssetFeatureExtractor,
    load_market_data,
)
from src.ml.triple_barrier import TripleBarrierLabeler, BarrierLabel
from src.ml.combinatorial_purged_cv import CombinatorialPurgedCV
from src.ml.gmm_regime import GMMRegimeDetector
from src.ml.drawdown_target import DrawdownTargetGenerator
from src.ml.frac_diff import FracDiff
from src.ml.historical_analog import HistoricalAnalogMatcher, AnalogMatchSet, AnalogResult

# Phase 3 — New Models (FS1, FS8, FS13, T5, T8)
from src.ml.survival_analyzer import (
    SurvivalAnalyzer,
    SurvivalResult,
    SurvivalTarget,
    compare_survival_models,
)
from src.ml.volatility_forecaster import VolatilityForecaster, VolForecastResult
from src.ml.garch_forecaster import GARCHForecaster, GARCHForecastResult
from src.ml.breakout_classifier import BreakoutClassifier, BreakoutResult
from src.ml.ensemble_models import (
    EnsembleBuilder,
    EnsembleResult,
    EnsembleMethod,
    TaskType,
    BaseModelConfig,
)
from src.ml.shap_dashboard import (
    SHAPDashboard,
    PredictionExplanation,
    GlobalImportance,
    quick_shap_analysis,
)

# Phase 4 — Advanced Pipeline (FS7, FS9, FS10, FS12)
from src.ml.change_point_regime import ChangePointRegimeDetector
from src.ml.volume_profile_cluster import VolumeProfileCluster
from src.ml.hdbscan_anomaly import HDBSCANAnomalyDetector
from src.ml.cross_symbol_cluster import CrossSymbolCluster

# Phase 6b — Pioneer Research (T9, FS19)
from src.ml.meta_labeler import MetaLabeler, MetaLabelResult, MetaLabelPrediction
from src.ml.gap_fill_predictor import (
    GapFillPredictor,
    GapFillPrediction,
    GapFillTrainingResult,
    GapDetection,
)

# Phase 4+ — Risk, Alpha Research & Ensemble (T6, QW2, RS1, RS2)
from src.ml.stop_loss_optimizer import (
    StopLossOptimizer,
    StopLossRecommendation,
    LabelGenerator,
)
from src.ml.ebm_alpha import (
    EBMAlphaPipeline,
    AlphaResult,
    AlphaFeature,
    AlphaShape,
)
from src.ml.dream_team_ensemble import (
    DreamTeamEnsemble,
    DreamTeamResult,
    DreamTeamConfig,
)

# C5: Circuit overfitting detection
from src.ml.circuit_overfit import CircuitOverfitDetector, CircuitOverfitResult

# C6: Adversarial overfitting detection
from src.ml.adversarial_overfit import AdversarialOverfitDetector, AdversarialOverfitResult

# A+B: Regime-adaptive ML
from src.ml.simple_regime import (
    SimpleTrendRegimeDetector,
    SimpleVolRegimeDetector,
    CombinedSimpleRegimeDetector,
)
from src.ml.regime_router import RegimeRouter

# Phase 17: HP Filter Return Forecasting (R4)
from src.ml.expected_returns import (
    hp_filter,
    hp_forecast,
    hp_expected_return,
    HPFilter,
    HP_LAMBDA_DAILY,
    HP_LAMBDA_WEEKLY,
    HP_LAMBDA_MONTHLY,
    HP_LAMBDA_QUARTERLY,
)

# Phase 17 P1: Liquidity Factor (R6) + MAD Pipeline (R8)
from src.ml.factor_features import (
    compute_cei,
    compute_liquidity_factor,
    compute_cei_dataframe,
    compute_amihud_illiquidity,
    compute_roll_spread,
    add_liquidity_features,
    CEI_WINDOW,
    compute_dtd,
    compute_dtd_dataframe,
    add_dtd_features,
)
from src.ml.preprocessing import (
    MADOutlierClipper,
    RankStandardizer,
    mad_clip,
    rank_standardize,
    mad_rank_pipeline,
    InstanceNormalizer,
    instance_normalize,
)

# Phase 17 P3: Factor Engine Wrapper (R14)
from src.ml.factor_engine import (
    FactorEngineWrapper,
    FactorEngineStatus,
    get_factor_engine,
    KNOWN_FACTORS,
)

# Phase 21 Q5: Model Validation
from src.ml.model_validation import (
    ModelValidator,
    ValidationReport,
    DriftResult,
    validate_from_arrays,
    compute_mre_gap,
    verify_causal_masking,
)

# Phase 21 D3: Structural Break Detection
from src.ml.structural_break import (
    StructuralBreakDetector,
    StructuralBreakResult,
    UnitRootResult,
    Breakpoint,
    adf_test,
    kpss_test,
    chow_test,
    bai_perron_test,
)

# Phase 21 D10: Rolling ARIMA+GARCH Hybrid Forecaster
from src.ml.arima_garch import (
    ARIMAForecaster,
    ARIMAGARCHForecaster,
    ARIMAForecast,
    ARIMAGARCHResult,
)

# Phase 21 D11: State Space Models
from src.ml.state_space import (
    LocalLinearTrend,
    StateSpaceDecomposer,
    KalmanSignal,
    SSMDecomposition,
)

# Phase 21 Block B: Fixed Income Models (B1 Nelson-Siegel, B4 Credit Spreads, B5 Rate Models)
from src.ml.fixed_income_models import (
    NelsonSiegel,
    NelsonSiegelResult,
    CreditSpreadGate,
    CreditSpreadResult,
    VasicekModel,
    CIRModel,
    RateModelResult,
    BondPricer,
    BondResult,
)

# Phase 21 D9: Kalman Hedge Ratios
from src.ml.kalman_hedge import (
    KalmanHedgeEstimator,
    KalmanHedgePair,
    KalmanHedgeResult,
    PortfolioHedgeEstimator,
)

# Phase 21 C10: Wavelet/FFT Signal Processing
from src.ml.wavelet_signals import (
    WaveletDenoiser,
    DenoiseResult,
    FFTCycleDetector,
    CycleResult,
    SignalDecomposer,
    DecompositionResult,
    FFTFilter,
)

# Phase 21 D6: VAR + Granger Causality
from src.ml.var_granger import (
    VARModel,
    VARResult,
    GrangerCausalityTest,
    GrangerResult,
    CrossAssetLeadLag,
    LeadLagResult,
)

# Phase 21 Block A: Options Pricing (A1-A7)
from src.ml.options_pricing import (
    BlackScholes,
    BlackScholesResult,
    BinomialTree,
    MonteCarloPricer,
    HestonModel,
    SABRModel,
    VolSurface,
    VolSurfaceSlice,
    compute_greeks,
    delta_hedge_ratio,
)

# Phase 21 B11: CDS Pricing
from src.ml.cds_pricing import (
    CDSPricer,
    CDSCurveResult,
    CDSSpread,
)

# Phase 21 A12: Options Visualization
from src.ml.options_visualization import (
    OptionsVisualizer,
    OptionLeg,
)

# Phase 21 B7: Rate Derivatives Pricing
from src.ml.rate_derivatives import (
    DiscountCurve,
    InterestRateSwap,
    IRSResult,
    SwaptionPricer,
    SwaptionResult,
    CapFloorPricer,
    CapFloorResult,
    RateDerivativeSignal,
    compute_forward_curve_from_df,
    swap_spread_signal,
)

# Phase 24 P0: Model Validation Extensions (P24-8, P24-11)
from src.ml.model_validation import (
    bootstrap_performance_ci,
    PerformanceCI,
    FeatureImportanceMonitor,
    FeatureImportanceSnapshot,
)

# Phase 24 P1: Overfitting Detection + Profit Mirage
from src.ml.overfitting_detector import (
    TrainingHistoryOverfitDetector,
    OverfittingDetection,
    quick_overfitting_check,
)
from src.ml.profit_mirage import (
    run_profit_mirage,
    MirageReport,
    MirageFeatureImpact,
)

# Phase 25: Anti-Overfitting Infrastructure (C1, C2, C13, C22)
from src.ml.lock_box import LockBox, create_lock_box, create_lock_box_chronological
from src.ml.nested_cv import NestedPurgedCV, NestedCVResult, leave_one_group_out_cv
from src.ml.blind_analysis import BlindAnalysisResult, run_blind_analysis
from src.ml.label_shuffling import LabelShufflingResult, run_label_shuffling_test

# Phase 25: SVM Regime Classifier (B34)
from src.ml.svm_regime import SVMRegimeClassifier, SVMRegimeResult

# Phase 27A: GAN Data Augmentation
from src.ml.gan_data_augmentation import (
    TTSGAN,
    TTSGenerator,
    TTSDiscriminator,
    augment_dataset,
    prepare_gan_samples,
)
from src.ml.gan_convergence import (
    GANConvergenceMonitor,
    compute_dtw_dedims,
    compute_wasserstein_distance,
)

# Phase 27C: TadGAN Anomaly Detection
from src.ml.anomaly_detection import TadGAN, prepare_tadgan_samples

__all__ = [
    # Part A infrastructure
    "ExperimentLogger",
    "PurgedKFold",
    "FeatureStore",
    "compute_ic",
    "compute_rank_ic",
    "compute_information_ratio",
    "ic_summary",
    "filter_features_by_ic",
    "ModelRegistry",
    "BacktestBridge",
    "MLBacktestResult",
    # Existing components
    "FeatureEngineer",
    "RegimeClassifier",
    "MLRegimeState",
    "RegimeDetectorBase",
    "RegimeSummary",
    "RegimeType",
    "HMMRegimeDetector",
    "PCAKMeansRegimeDetector",
    "R2RDRegimeDetector",
    "RegimeFolioDetector",
    "MacroRegimeDetector",
    "PathSignatureRegimeDetector",
    "SignalScorer",
    "ScoredSignal",
    "MLPipeline",
    "PipelineResult",
    # New ML models (2025 stack)
    "CatBoostForecaster",
    "ChronosForecaster",
    "FinCastForecaster",
    "xLSTMForecaster",
    # Additional components (scripts/tests)
    "ModelSelector",
    "ModelConfig",
    "ModelResult",
    "Recommendation",
    "PatternClassifier",
    "FeatureExtractor",
    "FeatureConfig",
    "FeatureSelector",
    "SelectionResult",
    "EnsembleRegimeDetector",
    "EnsembleStrategy",
    "CrossAssetFeatures",
    "prepare_cross_asset_data",
    "CrossAssetFeatureExtractor",
    "load_market_data",
    # Triple barrier + combinatorial purged CV
    "TripleBarrierLabeler",
    "BarrierLabel",
    "CombinatorialPurgedCV",
    # Phase 2 — Quick Wins (FS2, FS3, FS5, FS11)
    "GMMRegimeDetector",
    "DrawdownTargetGenerator",
    "FracDiff",
    "HistoricalAnalogMatcher",
    "AnalogMatchSet",
    "AnalogResult",
    # Phase 3 — New Models (FS1, FS8, FS13, T5, T8)
    "SurvivalAnalyzer",
    "SurvivalResult",
    "SurvivalTarget",
    "compare_survival_models",
    "VolatilityForecaster",
    "VolForecastResult",
    "GARCHForecaster",
    "GARCHForecastResult",
    "BreakoutClassifier",
    "BreakoutResult",
    "EnsembleBuilder",
    "EnsembleResult",
    "EnsembleMethod",
    "TaskType",
    "BaseModelConfig",
    "SHAPDashboard",
    "PredictionExplanation",
    "GlobalImportance",
    "quick_shap_analysis",
    # Phase 4 — Advanced Pipeline (FS7, FS9, FS10, FS12)
    "ChangePointRegimeDetector",
    "VolumeProfileCluster",
    "HDBSCANAnomalyDetector",
    "CrossSymbolCluster",
    # Phase 4+ — Risk, Alpha Research & Ensemble (T6, QW2, RS1, RS2)
    "StopLossOptimizer",
    "StopLossRecommendation",
    "LabelGenerator",
    "EBMAlphaPipeline",
    "AlphaResult",
    "AlphaFeature",
    "AlphaShape",
    "DreamTeamEnsemble",
    "DreamTeamResult",
    "DreamTeamConfig",
    # Phase 6b — Pioneer Research
    "MetaLabeler",
    "MetaLabelResult",
    "MetaLabelPrediction",
    "GapFillPredictor",
    "GapFillPrediction",
    "GapFillTrainingResult",
    "GapDetection",
    # C5: Circuit overfitting detection
    "CircuitOverfitDetector",
    "CircuitOverfitResult",
    # C6: Adversarial overfitting detection
    "AdversarialOverfitDetector",
    "AdversarialOverfitResult",
    # A+B: Regime-adaptive ML
    "SimpleTrendRegimeDetector",
    "SimpleVolRegimeDetector",
    "CombinedSimpleRegimeDetector",
    "RegimeRouter",
    # Phase 17: HP Filter (R4)
    "hp_filter",
    "hp_forecast",
    "hp_expected_return",
    "HPFilter",
    "HP_LAMBDA_DAILY",
    "HP_LAMBDA_WEEKLY",
    "HP_LAMBDA_MONTHLY",
    "HP_LAMBDA_QUARTERLY",
    # Phase 17 P1: Liquidity Factor (R6) + MAD Pipeline (R8)
    "compute_cei",
    "compute_liquidity_factor",
    "compute_cei_dataframe",
    "compute_amihud_illiquidity",
    "compute_roll_spread",
    "add_liquidity_features",
    "CEI_WINDOW",
    "MADOutlierClipper",
    "RankStandardizer",
    "mad_clip",
    "rank_standardize",
    "mad_rank_pipeline",
    "InstanceNormalizer",
    "instance_normalize",
    # Phase 17 P2: Default Risk Factor (R11)
    "compute_dtd",
    "compute_dtd_dataframe",
    "add_dtd_features",
    # Phase 17 P3: Factor Engine Wrapper (R14)
    "FactorEngineWrapper",
    "FactorEngineStatus",
    "get_factor_engine",
    "KNOWN_FACTORS",
    # Phase 21 Q5: Model Validation
    "ModelValidator",
    "ValidationReport",
    "DriftResult",
    "validate_from_arrays",
    # Phase 21 D3: Structural Break Detection
    "StructuralBreakDetector",
    "StructuralBreakResult",
    "UnitRootResult",
    "Breakpoint",
    "adf_test",
    "kpss_test",
    "chow_test",
    "bai_perron_test",
    # Phase 21 D10: Rolling ARIMA+GARCH Hybrid Forecaster
    "ARIMAForecaster",
    "ARIMAGARCHForecaster",
    "ARIMAForecast",
    "ARIMAGARCHResult",
    # Phase 21 D11: State Space Models
    "LocalLinearTrend",
    "StateSpaceDecomposer",
    "KalmanSignal",
    "SSMDecomposition",
    # Phase 21 Block B: Fixed Income Models
    "NelsonSiegel",
    "NelsonSiegelResult",
    "CreditSpreadGate",
    "CreditSpreadResult",
    "VasicekModel",
    "CIRModel",
    "RateModelResult",
    "BondPricer",
    "BondResult",
    # Phase 21 D9: Kalman Hedge Ratios
    "KalmanHedgeEstimator",
    "KalmanHedgePair",
    "KalmanHedgeResult",
    "PortfolioHedgeEstimator",
    # Phase 21 C10: Wavelet/FFT Signal Processing
    "WaveletDenoiser",
    "DenoiseResult",
    "FFTCycleDetector",
    "CycleResult",
    "SignalDecomposer",
    "DecompositionResult",
    "FFTFilter",
    # Phase 21 D6: VAR + Granger Causality
    "VARModel",
    "VARResult",
    "GrangerCausalityTest",
    "GrangerResult",
    "CrossAssetLeadLag",
    "LeadLagResult",
    # Phase 21 Block A: Options Pricing
    "BlackScholes",
    "BlackScholesResult",
    "BinomialTree",
    "MonteCarloPricer",
    "HestonModel",
    "SABRModel",
    "VolSurface",
    "VolSurfaceSlice",
    "compute_greeks",
    "delta_hedge_ratio",
    # Phase 21 B11: CDS Pricing
    "CDSPricer",
    "CDSCurveResult",
    "CDSSpread",
    # Phase 21 A12: Options Visualization
    "OptionsVisualizer",
    "OptionLeg",
    # Phase 21 B7: Rate Derivatives Pricing
    "DiscountCurve",
    "InterestRateSwap",
    "IRSResult",
    "SwaptionPricer",
    "SwaptionResult",
    "CapFloorPricer",
    "CapFloorResult",
    "RateDerivativeSignal",
    "compute_forward_curve_from_df",
    "swap_spread_signal",
    # Phase 24 P0: Model Validation Extensions (P24-8, P24-11)
    "bootstrap_performance_ci",
    "PerformanceCI",
    "FeatureImportanceMonitor",
    "FeatureImportanceSnapshot",
    # Phase 24 P1: Overfitting Detection + Profit Mirage
    "TrainingHistoryOverfitDetector",
    "OverfittingDetection",
    "quick_overfitting_check",
    "run_profit_mirage",
    "MirageReport",
    "MirageFeatureImpact",
    # Phase 25: Anti-Overfitting Infrastructure (C1, C2, C13, C22)
    "LockBox",
    "create_lock_box",
    "create_lock_box_chronological",
    "NestedPurgedCV",
    "NestedCVResult",
    "leave_one_group_out_cv",
    "BlindAnalysisResult",
    "run_blind_analysis",
    "LabelShufflingResult",
    "run_label_shuffling_test",
    # Phase 25: SVM Regime Classifier (B34)
    "SVMRegimeClassifier",
    "SVMRegimeResult",
    # Phase 27A: TTS-GAN Data Augmentation
    "TTSGAN",
    "TTSGenerator",
    "TTSDiscriminator",
    "prepare_gan_samples",
    "augment_dataset",
    "GANConvergenceMonitor",
    "compute_dtw_dedims",
    "compute_wasserstein_distance",
    # Phase 27C: TadGAN Anomaly Detection
    "TadGAN",
    "prepare_tadgan_samples",
]
