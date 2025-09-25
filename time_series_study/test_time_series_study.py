import numpy as np
import pandas as pd

from time_series_study import TimeSeriesStudy

try:
    import statsmodels  # type: ignore
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

def make_sample_df(n_groups: int = 5, n_dates: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    names = [f"G{i+1}" for i in range(n_groups)]

    index = pd.MultiIndex.from_product([dates, names], names=['date', 'name'])

    # Construct a few features with different dynamics
    # base signals per group
    base_levels = rng.normal(0.0, 1.0, size=n_groups)
    trends = rng.normal(0.01, 0.01, size=n_groups)
    seasonal_amp = rng.uniform(0.5, 1.5, size=n_groups)

    f1 = np.zeros(len(index))
    f2 = np.zeros(len(index))
    f3 = np.zeros(len(index))

    for gi, name in enumerate(names):
        # group slice indices
        rows = slice(gi, len(index), n_groups)
        t = np.arange(n_dates)
        # Feature 1: trend + noise
        f1_group = base_levels[gi] + trends[gi] * t + rng.normal(0, 0.3, size=n_dates)
        # Feature 2: seasonal + noise
        f2_group = seasonal_amp[gi] * np.sin(2 * np.pi * t / 12.0) + rng.normal(0, 0.2, size=n_dates)
        # Feature 3: AR(1)-like
        f3_group = np.zeros(n_dates)
        eps = rng.normal(0, 0.25, size=n_dates)
        phi = 0.6 + 0.2 * rng.random()
        for i in range(n_dates):
            f3_group[i] = (phi * f3_group[i-1] + eps[i]) if i > 0 else eps[i]

        f1[rows] = f1_group
        f2[rows] = f2_group
        f3[rows] = f3_group

    df = pd.DataFrame({'feature_1': f1, 'feature_2': f2, 'feature_3': f3}, index=index)
    return df


def main():
    df = make_sample_df()
    print("Input DataFrame (head):")
    print(df.head())
    print()

    study = TimeSeriesStudy(df)

    # 1) Distribution & Volatility Diagnostics
    dist_df = study.distribution_diagnostics(publish_plot=True, table=True)
    print("Distribution & Volatility Diagnostics (head):")
    print(dist_df.head())
    dist_df.to_csv('out_distribution.csv', index=False)
    print()

    # 2) Stationarity Tests (ADF, KPSS)
    if HAS_STATSMODELS:
        stat_df, stat_summary = study.stationarity_tests(regression='c', publish_plot=True, table=True)
        print("Stationarity Tests (head):")
        print(stat_df.head())
        print("Stationarity Summary:")
        print(stat_summary)
        stat_df.to_csv('out_stationarity.csv', index=False)
        stat_summary.to_csv('out_stationarity_summary.csv', index=False)
        print()
    else:
        print("[skip] statsmodels not installed: stationarity tests")
        print()

    # 3) Memory & Dependence Structure
    if HAS_STATSMODELS:
        mem_df = study.memory_dependence(publish_plot=True, table=True)
        print("Memory & Dependence (head):")
        print(mem_df.head())
        mem_df.to_csv('out_memory.csv', index=False)
        print()
    else:
        print("[skip] statsmodels not installed: memory & dependence")
        print()

    # 4) Frequency-Domain Analysis
    freq_df, _ = study.frequency_domain(detrend=True, publish_plot=True, table=True)
    print("Frequency-Domain (head):")
    print(freq_df.head())
    freq_df.to_csv('out_frequency.csv', index=False)
    print()

    # 5) Cross-Sectional Dispersion (one feature)
    cs_disp = study.cross_sectional_dispersion('feature_1', publish_plot=True, table=True)
    print("Cross-Sectional Dispersion (feature_1) (head):")
    print(cs_disp.head())
    cs_disp.to_frame().to_csv('out_cross_section_dispersion_feature1.csv')
    print()

    # 6) Predictability (AR(1) Fit)
    ar1_df = study.predictability_ar1(publish_plot=True, table=True)
    print("Predictability AR(1) (head):")
    print(ar1_df.head())
    ar1_df.to_csv('out_ar1.csv', index=False)
    print()

    # 7) Cross-Feature Comparisons
    if HAS_STATSMODELS:
        pairs = [('feature_1', 'feature_2'), ('feature_1', 'feature_3')]
        xfeat = study.cross_feature_comparisons(feature_pairs=pairs, max_lag=6, rolling_window=10, plot=False, publish_plot=True, table=True)
        print("Lagged Correlations (head):")
        print(xfeat['lagged_corr'].head())
        xfeat['lagged_corr'].to_csv('out_lagged_corr.csv')

        print("Cointegration (head):")
        print(xfeat['cointegration'].head())
        xfeat['cointegration'].to_csv('out_cointegration.csv', index=False)

        print("Granger (head):")
        print(xfeat['granger'].head())
        xfeat['granger'].to_csv('out_granger.csv', index=False)

        print("Mutual Information (head):")
        print(xfeat['mutual_info'].head())
        xfeat['mutual_info'].to_csv('out_mutual_info.csv', index=False)
    else:
        print("[skip] statsmodels not installed: cross-feature comparisons")

    print("\nAll outputs saved to CSV files with prefix 'out_'.")
    print()

    # 8) Advanced Analysis Methods
    print("=" * 60)
    print("ADVANCED ANALYSIS METHODS")
    print("=" * 60)

    # Regime Detection
    if HAS_STATSMODELS:
        print("8) Regime Detection (Markov Switching)")
        regime_results = study.regime_detection('feature_1', n_regimes=2, publish_plot=True, table=True)
        print("Regime Detection Summary:")
        print(f"  Average switches: {regime_results['summary']['avg_switches']:.2f}")
        print(f"  Average regime 1 prob: {regime_results['summary']['avg_regime_1_prob']:.3f}")
        print(f"  Average regime 2 prob: {regime_results['summary']['avg_regime_2_prob']:.3f}")
        print()

    # Volatility Analysis
    if HAS_STATSMODELS:
        print("9) Volatility Analysis (GARCH)")
        vol_results = study.volatility_analysis('feature_1', model_type='GARCH', publish_plot=True, table=True)
        print("Volatility Analysis Summary:")
        print(f"  Average persistence: {vol_results['summary']['avg_persistence']:.3f}")
        print(f"  Average volatility: {vol_results['summary']['avg_volatility']:.3f}")
        print(f"  Volatility clustering: {vol_results['summary']['vol_clustering']}")
        print()

    # Outlier Detection
    print("10) Outlier Detection")
    outlier_results = study.outlier_detection('feature_1', method='isolation_forest', publish_plot=True, table=True)
    print("Outlier Detection Summary:")
    print(f"  Average outlier rate: {outlier_results['summary']['avg_outlier_rate']:.3f}")
    print(f"  Groups with outliers: {outlier_results['summary']['groups_with_outliers']}")
    print(f"  Total outliers: {outlier_results['summary']['total_outliers']}")
    print()

    print("Advanced analysis complete! Check time_series_study/results/ for all outputs.")


if __name__ == '__main__':
    main()


