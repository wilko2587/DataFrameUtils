import itertools
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import os


@dataclass
class GroupKey:
    """Container for a group key derived from a MultiIndex (excluding 'date')."""
    values: Tuple[Any, ...]

    def as_tuple(self) -> Tuple[Any, ...]:
        return self.values

    def __str__(self) -> str:
        return ":".join(map(str, self.values)) if self.values else "__ALL__"


class TimeSeriesStudy:
    """
    TimeSeriesStudy(df)

    Analyze multi-group time series stored in a pandas DataFrame with a MultiIndex.

    Requirements
    - The DataFrame index must be a MultiIndex and include a level named 'date'.
    - All non-index columns are considered features (float-like).
    - Other index levels (besides 'date') are used purely for grouping.

    Example shape
    If index names are ['date', 'name'] with 10 dates and 5 names, the DataFrame
    will have 50 rows. Each column represents a feature containing 5 separate
    time series (one per 'name'), each of length 10.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = self._validate_and_prepare(df.copy())
        self.index_names: List[str] = list(self.df.index.names)
        self.date_level: str = 'date'
        self.group_levels: List[str] = [n for n in self.index_names if n != self.date_level]
        self.features: List[str] = [c for c in self.df.columns]
        # default results directory
        self.default_results_dir = os.path.join('time_series_study', 'results')

    @staticmethod
    def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame index must be a MultiIndex including a 'date' level.")
        if 'date' not in df.index.names:
            raise ValueError("MultiIndex must contain a level named 'date'.")
        # Ensure date level is datetime-like and sorted within groups
        if not np.issubdtype(pd.Series(df.index.get_level_values('date')).dtype, np.datetime64):
            try:
                df = df.copy()
                date_values = pd.to_datetime(df.index.get_level_values('date'))
                df.index = df.index.set_levels(
                    [pd.to_datetime(lv) if name == 'date' else lv
                     for lv, name in zip(df.index.levels, df.index.names)]
                )
            except Exception as exc:
                raise ValueError("'date' index level must be datetime-like or convertible.") from exc

        # Sort by full index to guarantee chronological order within groups
        df = df.sort_index()

        # Validate feature dtypes are numeric
        non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"All feature columns must be numeric. Non-numeric: {non_numeric}")

        return df

    def _iter_groups(self) -> Iterable[Tuple[GroupKey, pd.DataFrame]]:
        """
        Iterate over groups determined by all index levels except 'date'.
        If there are no grouping levels (i.e., only 'date'), yield a single group
        with an empty GroupKey.
        """
        if len(self.group_levels) == 0:
            yield GroupKey(tuple()), self.df
            return

        for key_vals, subdf in self.df.groupby(level=self.group_levels, sort=False):
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            yield GroupKey(tuple(key_vals)), subdf

    # ---------- Presentation helpers ----------
    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _minimalist_axes(ax, title: Optional[str] = None) -> None:
        # Ultra-minimalist aesthetic: no spines, no grid, sparse ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_facecolor('white')
        if title:
            ax.set_title(title, fontsize=12, pad=8)
        ax.tick_params(axis='both', which='both', length=0, labelsize=9)

    @staticmethod
    def _palette() -> List[str]:
        # Consistent color order: black, red, then complementary set
        return ['#000000', '#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD', '#8C564B']

    @staticmethod
    def _save_df_table(df: pd.DataFrame, out_path_no_ext: str) -> None:
        # Save both CSV and simple styled HTML
        csv_path = out_path_no_ext + '.csv'
        html_path = out_path_no_ext + '.html'
        df.to_csv(csv_path, index=False)
        try:
            styled = df.head(1000).style.set_table_styles([
                {'selector': 'th', 'props': [('font-weight', '600'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('padding', '4px 8px')]},
            ]).hide_index()
            styled.to_html(html_path)
        except Exception:
            pass

    # 1) Distribution & Volatility Diagnostics
    def distribution_diagnostics(self, publish_plot: bool = False, table: bool = False, results_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Summarize the distributional shape of each feature within each group.

        For each (group × feature) compute: mean, standard deviation, skewness, kurtosis.

        Returns
        - DataFrame with columns: ['group', 'feature', 'mean', 'std', 'skew', 'kurtosis']
          where 'group' is a string key constructed from non-date index values (or '__ALL__').

        If plot=True, shows histograms of skewness and kurtosis across groups per feature.
        """
        records: List[Dict[str, Any]] = []
        for gkey, subdf in self._iter_groups():
            # Collapse to time index only
            ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
            for feature in self.features:
                series = ts_df[feature].dropna()
                if len(series) == 0:
                    continue
                records.append({
                    'group': str(gkey),
                    'feature': feature,
                    'mean': float(series.mean()),
                    'std': float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                    'skew': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                })

        result = pd.DataFrame.from_records(records)

        if table and not result.empty:
            out_dir = results_dir or self.default_results_dir
            self._ensure_dir(out_dir)
            self._save_df_table(result, os.path.join(out_dir, 'distribution_diagnostics'))

        if publish_plot and not result.empty:
            import matplotlib.pyplot as plt
            plt.ioff()
            palette = self._palette()
            out_dir = results_dir or self.default_results_dir
            self._ensure_dir(out_dir)
            for feature, fdf in result.groupby('feature'):
                fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                axes[0].hist(fdf['skew'].dropna(), bins=16, color=palette[1])
                self._minimalist_axes(axes[0], title=f"Skew: {feature}")
                axes[1].hist(fdf['kurtosis'].dropna(), bins=16, color=palette[2])
                self._minimalist_axes(axes[1], title=f"Kurtosis: {feature}")
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"distribution_{feature}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)

        return result.sort_values(['feature', 'group']).reset_index(drop=True)

    # 2) Stationarity Tests (ADF, KPSS)
    def stationarity_tests(self, regression: str = 'c', publish_plot: bool = False, table: bool = False, results_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Assess stationarity for each (group × feature) via ADF and KPSS tests.

        Parameters
        - regression: 'c' (constant) or 'ct' (constant + trend)

        Returns
        - results_df: columns = ['group','feature','adf_stat','adf_p','kpss_stat','kpss_p']
        - summary_df: per-feature proportions rejecting each null

        Notes
        - ADF null: unit root (non-stationary). Low p rejects unit root.
        - KPSS null: stationary. Low p rejects stationarity.
        """
        from statsmodels.tsa.stattools import adfuller, kpss

        records: List[Dict[str, Any]] = []
        for gkey, subdf in self._iter_groups():
            ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
            for feature in self.features:
                x = ts_df[feature].dropna().astype(float).values
                if len(x) < 10:
                    continue
                try:
                    adf_stat, adf_p, *_ = adfuller(x, autolag='AIC', regression=('ct' if regression=='ct' else 'c'))
                except Exception:
                    adf_stat, adf_p = np.nan, np.nan
                try:
                    kpss_stat, kpss_p, *_ = kpss(x, regression=('ct' if regression=='ct' else 'c'), nlags='auto')
                except Exception:
                    kpss_stat, kpss_p = np.nan, np.nan
                records.append({
                    'group': str(gkey),
                    'feature': feature,
                    'adf_stat': adf_stat,
                    'adf_p': adf_p,
                    'kpss_stat': kpss_stat,
                    'kpss_p': kpss_p,
                })

        results_df = pd.DataFrame.from_records(records)
        if results_df.empty:
            return results_df, pd.DataFrame(columns=['feature','prop_reject_unit_root','prop_reject_stationarity'])

        summaries: List[Dict[str, Any]] = []
        for feature, fdf in results_df.groupby('feature'):
            prop_reject_unit_root = float((fdf['adf_p'] < 0.05).mean())
            prop_reject_stationarity = float((fdf['kpss_p'] < 0.05).mean())
            summaries.append({
                'feature': feature,
                'prop_reject_unit_root': prop_reject_unit_root,
                'prop_reject_stationarity': prop_reject_stationarity,
            })
        summary_df = pd.DataFrame.from_records(summaries)
        results_df = results_df.sort_values(['feature','group']).reset_index(drop=True)
        summary_df = summary_df.sort_values('feature').reset_index(drop=True)

        out_dir = results_dir or self.default_results_dir
        if table and not results_df.empty:
            self._ensure_dir(out_dir)
            self._save_df_table(results_df, os.path.join(out_dir, 'stationarity_results'))
            self._save_df_table(summary_df, os.path.join(out_dir, 'stationarity_summary'))

        if publish_plot and not results_df.empty:
            import matplotlib.pyplot as plt
            plt.ioff()
            palette = self._palette()
            self._ensure_dir(out_dir)
            # Proportion bars per feature
            fig, ax = plt.subplots(figsize=(6, 3))
            x = np.arange(len(summary_df))
            w = 0.35
            ax.bar(x - w/2, summary_df['prop_reject_unit_root'].values, width=w, color=palette[1], label='ADF rejects')
            ax.bar(x + w/2, summary_df['prop_reject_stationarity'].values, width=w, color=palette[2], label='KPSS rejects')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['feature'].values, rotation=0, fontsize=9)
            self._minimalist_axes(ax, title='Stationarity rejections')
            ax.legend(frameon=False, fontsize=8, loc='upper right')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, 'stationarity_summary.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Dumbbell-like: per feature, sorted by group index; plot lines ADF vs KPSS p
            for feature, fdf in results_df.groupby('feature'):
                if fdf.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                y = np.arange(len(fdf))
                a = fdf['adf_p'].values
                k = fdf['kpss_p'].values
                for i in range(len(y)):
                    ax.plot([a[i], k[i]], [y[i], y[i]], color=palette[0], linewidth=2)
                ax.scatter(a, y, color=palette[1], s=16, label='ADF p')
                ax.scatter(k, y, color=palette[2], s=16, label='KPSS p')
                ax.set_xlabel('p-value')
                ax.set_yticks([])
                self._minimalist_axes(ax, title=f'Stationarity p-values: {feature}')
                ax.legend(frameon=False, fontsize=8, loc='lower right')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f'stationarity_dumbbell_{feature}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)

        return results_df, summary_df

    # 3) Memory & Dependence Structure
    def memory_dependence(self, publish_plot: bool = False, table: bool = False, results_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Quantify dependence on the past per (group × feature).

        Computes
        - Autocorrelation at lags 1, 5, 10
        - Ljung–Box test p-value at max lag 10
        - Hurst exponent (rescaled range estimate)

        Returns
        - DataFrame with columns: ['group','feature','acf_lag1','acf_lag5','acf_lag10','ljungbox_p','hurst_exp']
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        def hurst_rs(series: np.ndarray) -> float:
            x = np.asarray(series, dtype=float)
            n = len(x)
            if n < 20:
                return np.nan
            mean_x = x.mean()
            y = np.cumsum(x - mean_x)
            r = y.max() - y.min()
            s = x.std(ddof=1)
            if s == 0:
                return 0.5
            rs = r / s
            return float(np.log(rs) / np.log(n))

        records: List[Dict[str, Any]] = []
        for gkey, subdf in self._iter_groups():
            ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
            for feature in self.features:
                s = ts_df[feature].dropna().astype(float)
                if len(s) < 12:
                    continue
                acf_lag1 = float(s.autocorr(lag=1))
                acf_lag5 = float(s.autocorr(lag=5)) if len(s) > 5 else np.nan
                acf_lag10 = float(s.autocorr(lag=10)) if len(s) > 10 else np.nan
                try:
                    lb = acorr_ljungbox(s, lags=[10], return_df=True)
                    ljungbox_p = float(lb['lb_pvalue'].iloc[-1])
                except Exception:
                    ljungbox_p = np.nan
                h = hurst_rs(s.values)
                records.append({
                    'group': str(gkey),
                    'feature': feature,
                    'acf_lag1': acf_lag1,
                    'acf_lag5': acf_lag5,
                    'acf_lag10': acf_lag10,
                    'ljungbox_p': ljungbox_p,
                    'hurst_exp': h,
                })

        df_out = pd.DataFrame.from_records(records).sort_values(['feature','group']).reset_index(drop=True)

        out_dir = results_dir or self.default_results_dir
        if table and not df_out.empty:
            self._ensure_dir(out_dir)
            self._save_df_table(df_out, os.path.join(out_dir, 'memory_dependence'))

        if publish_plot and not df_out.empty:
            import matplotlib.pyplot as plt
            plt.ioff()
            palette = self._palette()
            self._ensure_dir(out_dir)
            # Hurst histogram per feature
            for feature, fdf in df_out.groupby('feature'):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(fdf['hurst_exp'].dropna(), bins=16, color=palette[2])
                self._minimalist_axes(ax, title=f'Hurst exponent: {feature}')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f'hurst_{feature}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)

        return df_out

    # 4) Frequency-Domain Analysis
    def frequency_domain(self, detrend: bool = True, publish_plot: bool = False, table: bool = False, results_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """
        Detect cyclic structure using the (discrete) periodogram per (group × feature).

        For each series, computes the periodogram and extracts the frequency with
        maximum power (excluding zero freq if possible).

        Returns
        - results_df: ['group','feature','dom_freq','dom_power']
        - summary (optional): dict (reserved for future enhancements)
        """
        try:
            from scipy.signal import periodogram
        except Exception:
            periodogram = None

        results: List[Dict[str, Any]] = []
        for gkey, subdf in self._iter_groups():
            ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
            for feature in self.features:
                x = ts_df[feature].dropna().astype(float).values
                if len(x) < 8:
                    continue
                if detrend:
                    t = np.arange(len(x))
                    coeffs = np.polyfit(t, x, 1)
                    x = x - (coeffs[0] * t + coeffs[1])
                if periodogram is not None:
                    freqs, power = periodogram(x)
                else:
                    fx = np.fft.rfft(x)
                    power = (fx.real**2 + fx.imag**2)
                    freqs = np.fft.rfftfreq(len(x), d=1.0)
                if len(freqs) == 0:
                    continue
                start = 1 if len(freqs) > 1 else 0
                idx = start + int(np.argmax(power[start:]))
                results.append({
                    'group': str(gkey),
                    'feature': feature,
                    'dom_freq': float(freqs[idx]),
                    'dom_power': float(power[idx]),
                })

        df_out = pd.DataFrame.from_records(results).sort_values(['feature','group']).reset_index(drop=True)

        out_dir = results_dir or self.default_results_dir
        if table and not df_out.empty:
            self._ensure_dir(out_dir)
            self._save_df_table(df_out, os.path.join(out_dir, 'frequency_domain'))

        if publish_plot and not df_out.empty:
            import matplotlib.pyplot as plt
            plt.ioff()
            palette = self._palette()
            self._ensure_dir(out_dir)
            for feature, fdf in df_out.groupby('feature'):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.scatter(fdf['dom_freq'], fdf['dom_power'], color=palette[1], s=14)
                ax.set_xlabel('dom freq')
                ax.set_ylabel('power')
                self._minimalist_axes(ax, title=f'Dominant frequency: {feature}')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f'dom_freq_{feature}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)

        return df_out, None


    # 5) Cross-Sectional Dispersion
    def cross_sectional_dispersion(self, feature: str, publish_plot: bool = False, table: bool = False, results_dir: Optional[str] = None) -> pd.Series:
        """
        Show how much groups diverge from each other at each date for a given feature.

        For each date, compute the standard deviation across groups.

        Returns
        - Series indexed by date with cross-sectional dispersion values.
        """
        if feature not in self.features:
            raise ValueError(f"Unknown feature: {feature}")
        if len(self.group_levels) == 0:
            s = self.df.droplevel([])[feature]
            cs = pd.Series(0.0, index=s.index, name=f"dispersion_{feature}")
        else:
            wide = self.df[feature].unstack(self.group_levels)
            cs = wide.std(axis=1)
            cs.name = f"dispersion_{feature}"

        out_dir = results_dir or self.default_results_dir
        if table:
            self._ensure_dir(out_dir)
            self._save_df_table(cs.reset_index().rename(columns={0: cs.name, cs.name: 'dispersion'}), os.path.join(out_dir, f'dispersion_{feature}'))

        if publish_plot:
            import matplotlib.pyplot as plt
            plt.ioff()
            self._ensure_dir(out_dir)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(cs.index, cs.values, color=self._palette()[0], linewidth=2)
            self._minimalist_axes(ax, title=f'Dispersion: {feature}')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'dispersion_{feature}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

        return cs

    # 6) Predictability (AR(1) Fit)
    def predictability_ar1(self, publish_plot: bool = False, table: bool = False, results_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Quick test of linear predictability: fit x[t] on x[t-1] per (group × feature).

        Returns
        - DataFrame with columns: ['group','feature','ar1_r2'] with R^2 per series.
        """
        records: List[Dict[str, Any]] = []
        for gkey, subdf in self._iter_groups():
            ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
            for feature in self.features:
                s = ts_df[feature].astype(float)
                y = s.shift(0).iloc[1:].values
                x = s.shift(1).iloc[1:].values
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if len(y) < 10:
                    continue
                X = np.vstack([np.ones_like(x), x]).T
                try:
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    yhat = X @ beta
                    ss_res = float(np.sum((y - yhat) ** 2))
                    ss_tot = float(np.sum((y - y.mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                except Exception:
                    r2 = np.nan
                records.append({'group': str(gkey), 'feature': feature, 'ar1_r2': r2})
        df_out = pd.DataFrame.from_records(records).sort_values(['feature','group']).reset_index(drop=True)

        out_dir = results_dir or self.default_results_dir
        if table and not df_out.empty:
            self._ensure_dir(out_dir)
            self._save_df_table(df_out, os.path.join(out_dir, 'predictability_ar1'))

        if publish_plot and not df_out.empty:
            import matplotlib.pyplot as plt
            plt.ioff()
            palette = self._palette()
            self._ensure_dir(out_dir)
            for feature, fdf in df_out.groupby('feature'):
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(fdf['ar1_r2'].dropna(), bins=16, color=palette[1])
                self._minimalist_axes(ax, title=f'AR(1) R²: {feature}')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f'ar1_r2_{feature}.png'), dpi=150, bbox_inches='tight')
                plt.close(fig)

        return df_out

    # 7) Cross-Feature Comparisons (within same groups)
    def cross_feature_comparisons(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        max_lag: int = 10,
        rolling_window: int = 10,
        example_group: Optional[Tuple[Any, ...]] = None,
        plot: bool = False,
        publish_plot: bool = False,
        table: bool = False,
        results_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Study relationships between pairs of features within the same groups.

        For each pair (feature_a, feature_b), per group:
        - Lagged correlations: Corr(a[t-L], b[t]) for L=0..max_lag. Returns cross-group averages.
        - Rolling correlation: for one example group, compute rolling window correlation (plot if requested).
        - Cointegration (Engle–Granger): statistic and p-value per group.
        - Granger causality a→b: minimum p-value across lags up to max_lag per group.
        - Mutual information (discretized): scalar per group.

        Returns dict with DataFrames for 'lagged_corr', 'cointegration', 'granger', 'mutual_info'.
        """
        from statsmodels.tsa.stattools import coint, grangercausalitytests

        if feature_pairs is None:
            feature_pairs = list(itertools.combinations(self.features, 2))

        def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            n = min(np.isfinite(x).sum(), np.isfinite(y).sum())
            if n < 10:
                return np.nan
            x = x[np.isfinite(x)][-n:]
            y = y[np.isfinite(y)][-n:]
            cxy, _, _ = np.histogram2d(x, y, bins=bins)
            pxy = cxy / np.sum(cxy)
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                mi = pxy * (np.log(pxy + 1e-12) - np.log(px + 1e-12) - np.log(py + 1e-12))
            return float(np.nansum(mi))

        lag_rows: List[Dict[str, Any]] = []
        coint_rows: List[Dict[str, Any]] = []
        granger_rows: List[Dict[str, Any]] = []
        mi_rows: List[Dict[str, Any]] = []

        # Choose example group for rolling correlation demo
        example_key = None
        if example_group is not None:
            example_key = GroupKey(tuple(example_group))
        else:
            for gkey, _ in self._iter_groups():
                example_key = gkey
                break

        # Lagged correlations averaged across groups
        for (fa, fb) in feature_pairs:
            lag_corrs = {lag: [] for lag in range(max_lag + 1)}
            for gkey, subdf in self._iter_groups():
                ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
                a = ts_df[fa].astype(float)
                b = ts_df[fb].astype(float)
                for lag in range(0, max_lag + 1):
                    if lag > 0:
                        ac = a.shift(lag)
                        aligned = pd.concat([ac, b], axis=1).dropna()
                    else:
                        aligned = pd.concat([a, b], axis=1).dropna()
                    if len(aligned) < 10:
                        continue
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    lag_corrs[lag].append(corr)
            for lag, vals in lag_corrs.items():
                lag_rows.append({'pair': f'{fa}|{fb}', 'lag': lag, 'avg_corr': np.nanmean(vals) if len(vals) else np.nan})

        lagged_corr_df = pd.DataFrame.from_records(lag_rows).pivot(index='lag', columns='pair', values='avg_corr') if lag_rows else pd.DataFrame()

        # Cointegration, Granger causality, Mutual information per group
        for (fa, fb) in feature_pairs:
            for gkey, subdf in self._iter_groups():
                ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
                a = ts_df[fa].astype(float).dropna()
                b = ts_df[fb].astype(float).dropna()
                aligned = pd.concat([a, b], axis=1, join='inner').dropna()
                if len(aligned) < max(20, max_lag + 5):
                    c_stat = c_p = g_p = np.nan
                    mi_val = np.nan
                else:
                    try:
                        c_stat, c_p, _ = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    except Exception:
                        c_stat, c_p = np.nan, np.nan
                    try:
                        gc_res = grangercausalitytests(aligned[[fa, fb]], maxlag=min(max_lag, 5), verbose=False)
                        g_p = float(min(v[0]['ssr_chi2test'][1] for _, v in gc_res.items()))
                    except Exception:
                        g_p = np.nan
                    mi_val = mutual_information(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values, bins=10)
                coint_rows.append({'group': str(gkey), 'pair': f'{fa}|{fb}', 'coint_stat': c_stat, 'coint_p': c_p})
                granger_rows.append({'group': str(gkey), 'pair': f'{fa}|{fb}', 'pvalue': g_p})
                mi_rows.append({'group': str(gkey), 'pair': f'{fa}|{fb}', 'mi': mi_val})

        # Optional tables
        out_dir = results_dir or self.default_results_dir
        if table:
            self._ensure_dir(out_dir)
            if not lagged_corr_df.empty:
                lagged_corr_df.to_csv(os.path.join(out_dir, 'lagged_corr.csv'))
            if coint_rows:
                pd.DataFrame.from_records(coint_rows).to_csv(os.path.join(out_dir, 'cointegration.csv'), index=False)
            if granger_rows:
                pd.DataFrame.from_records(granger_rows).to_csv(os.path.join(out_dir, 'granger.csv'), index=False)
            if mi_rows:
                pd.DataFrame.from_records(mi_rows).to_csv(os.path.join(out_dir, 'mutual_info.csv'), index=False)

        # Optional plots
        if (plot or publish_plot) and not lagged_corr_df.empty and example_key is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            palette = self._palette()
            self._ensure_dir(out_dir)
            # Lagged correlation heatmap-like line plot
            fig, ax = plt.subplots(figsize=(8, 3))
            for i, col in enumerate(lagged_corr_df.columns[:5]):
                ax.plot(lagged_corr_df.index, lagged_corr_df[col].values, linewidth=2, color=palette[i % len(palette)], label=col)
            self._minimalist_axes(ax, title='Avg lagged correlations (top 5 pairs)')
            ax.legend(frameon=False, fontsize=7, loc='upper right', ncol=1)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, 'lagged_corr.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Rolling correlation example for first pair and example group
            (fa, fb) = feature_pairs[0] if feature_pairs else (None, None)
            if fa and fb:
                for gkey, subdf in self._iter_groups():
                    if str(gkey) == str(example_key):
                        ts_df = subdf.droplevel(self.group_levels) if len(self.group_levels) else subdf
                        a = ts_df[fa].astype(float)
                        b = ts_df[fb].astype(float)
                        roll = pd.concat([a, b], axis=1).rolling(rolling_window).corr().unstack().iloc[:, 1]
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(roll.index, roll.values, color=palette[1], linewidth=2, label=f'{fa}|{fb}')
                        self._minimalist_axes(ax, title=f'Rolling corr ({fa} vs {fb}) - {gkey}')
                        ax.legend(frameon=False, fontsize=7, loc='upper right')
                        fig.tight_layout()
                        fig.savefig(os.path.join(out_dir, 'rolling_corr_example.png'), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        break

        out = {
            'lagged_corr': lagged_corr_df,
            'cointegration': pd.DataFrame.from_records(coint_rows).sort_values(['pair','group']).reset_index(drop=True),
            'granger': pd.DataFrame.from_records(granger_rows).sort_values(['pair','group']).reset_index(drop=True),
            'mutual_info': pd.DataFrame.from_records(mi_rows).sort_values(['pair','group']).reset_index(drop=True),
        }
        return out
