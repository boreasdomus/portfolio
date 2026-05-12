import argparse
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import utils

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimizer with two strategies:
    - MinVol: Minimum variance portfolio (default)
    - CAPM: Maximum Sharpe ratio using CAPM expected returns

    Reads portfolio tickers from portfolio.csv, loads price data from data/,
    and optimizes weights subject to constraints.
    """

    def __init__(
        self,
        data_dir="data",
        portfolio_file="portfolio.csv",
        benchmark_ticker="^OMXSBCAPGI",
        strategy="minvol",
        annual_rf=0.02,
        max_weight=0.15,
        min_weight=0.00,
        max_years=5,
        min_observations=50,
        correlation_threshold=0.95,
        data_coverage_threshold=0.7,
        trading_days=252,
        sector_filter=None,
    ):
        self.data_dir = data_dir
        self.portfolio_file = portfolio_file
        self.benchmark_ticker = benchmark_ticker
        self.strategy = strategy
        self.annual_rf = annual_rf
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_years = max_years
        self.min_observations = min_observations
        self.correlation_threshold = correlation_threshold
        self.data_coverage_threshold = data_coverage_threshold
        self.trading_days = trading_days
        self.sector_filter = sector_filter

        self.sector_map = {}
        self.prices_df = None
        self.benchmark_prices = None
        self.returns_df = None
        self.benchmark_returns = None
        self.results = {}

    def load_data(self):
        """Load portfolio symbols, price data, and benchmark."""
        symbols = self._load_portfolio_symbols()
        if not symbols:
            return

        if self.sector_filter:
            symbols = [s for s in symbols
                       if utils.matches_sector_filter(s, self.sector_map, self.sector_filter)]
            if not symbols:
                print(f"No symbols match sector filter '{self.sector_filter}'.")
                return
            print(f"Sector filter '{self.sector_filter}': {len(symbols)} symbols.")

        self.prices_df = utils.load_price_data(self.data_dir, symbols=symbols)
        if self.prices_df.empty:
            print("Error: No asset data loaded.")
            return

        self.benchmark_prices = utils.load_close_series(self.data_dir, self.benchmark_ticker)
        if self.benchmark_prices is None:
            print(f"Error: Benchmark file not found for {self.benchmark_ticker}")
            return

        # Align dates
        common_dates = self.prices_df.index.intersection(self.benchmark_prices.index)
        self.prices_df = self.prices_df.loc[common_dates]
        self.benchmark_prices = self.benchmark_prices.loc[common_dates]

        print(f"Data loaded: {len(self.prices_df.columns)} assets, "
              f"benchmark: {self.benchmark_ticker}")

    def _load_portfolio_symbols(self):
        """Load ticker list from portfolio.csv."""
        if not os.path.exists(self.portfolio_file):
            print(f"Error: Portfolio file not found: {self.portfolio_file}")
            print(f"Create {self.portfolio_file} with a 'Symbol' column.")
            return []

        try:
            df = pd.read_csv(self.portfolio_file)
            if "Symbol" not in df.columns:
                print(f"Error: Column 'Symbol' missing in {self.portfolio_file}")
                return []
            symbols = df["Symbol"].str.strip().tolist()
            print(f"Loaded {len(symbols)} symbols from {self.portfolio_file}")
            return symbols
        except Exception as e:
            print(f"Error loading {self.portfolio_file}: {e}")
            return []

    def _calculate_returns(self):
        """Calculate daily returns for assets and benchmark, align and trim."""
        print(f"\nCalculating returns...")

        # Daily returns
        asset_returns = self.prices_df.pct_change().dropna(how="all")
        bench_returns = self.benchmark_prices.pct_change().dropna()

        # Trim to max_years
        latest_date = asset_returns.index.max()
        cutoff_date = latest_date - pd.DateOffset(years=self.max_years)
        asset_returns = asset_returns[asset_returns.index >= cutoff_date]
        bench_returns = bench_returns[bench_returns.index >= cutoff_date]

        # Add benchmark as column for alignment
        combined = asset_returns.copy()
        combined["BENCHMARK"] = bench_returns

        # Remove assets with poor data coverage
        asset_cols = [c for c in combined.columns if c != "BENCHMARK"]
        coverage = {t: combined[t].notna().sum() / len(combined) for t in asset_cols}
        good_assets = [t for t, cov in coverage.items()
                       if cov >= self.data_coverage_threshold]

        removed = set(asset_cols) - set(good_assets)
        if removed:
            print(f"  Removed (poor coverage): {', '.join(removed)}")

        combined = combined[good_assets + ["BENCHMARK"]]
        combined.dropna(inplace=True)

        if len(combined) < self.min_observations:
            print(f"Error: Insufficient data ({len(combined)} < {self.min_observations})")
            return False

        self.returns_df = combined[good_assets]
        self.benchmark_returns = combined["BENCHMARK"]

        print(f"  Period: {combined.index.min().date()} to {combined.index.max().date()}")
        print(f"  {len(combined)} trading days, {len(good_assets)} assets")
        return True

    def run_optimizer(self):
        """Run the full optimization pipeline."""
        if self.prices_df is None:
            print("No data loaded. Run load_data() first.")
            return

        if not self._calculate_returns():
            return

        tickers = list(self.returns_df.columns)
        n_assets = len(tickers)

        if n_assets == 0:
            print("No assets available for optimization.")
            return

        # Correlation analysis
        self._analyze_correlations(tickers)

        # CAPM expected returns (if needed)
        daily_rf = (1 + self.annual_rf) ** (1 / self.trading_days) - 1
        expected_returns = None
        if self.strategy == "capm":
            expected_returns = self._calculate_capm_returns(tickers, daily_rf)
            if not expected_returns:
                print("Error: Could not calculate CAPM expected returns.")
                return

        strategy_names = {
            "minvol": "Minimum Variance",
            "capm": "CAPM (Max Sharpe)",
            "riskparity": "Risk Parity",
        }
        print(f"\nOptimizing: {strategy_names[self.strategy]}...")

        weights = self._optimize(tickers, expected_returns, daily_rf)

        # Drop positions below min_weight, then re-optimize on the subset so
        # max_weight stays binding (renormalizing after dropping breaks the cap).
        if self.min_weight > 0.0:
            kept = [t for t, w in zip(tickers, weights) if w >= self.min_weight]
            n_dropped = len(tickers) - len(kept)
            if n_dropped > 0 and len(kept) > 0:
                if self.max_weight * len(kept) < 1.0:
                    print(f"  Min-weight filter would drop {n_dropped} positions, "
                          f"but {len(kept)} remaining cannot sum to 1 under "
                          f"max_weight {self.max_weight:.1%}. Skipping filter.")
                else:
                    print(f"  Min-weight filter: dropped {n_dropped} positions below "
                          f"{self.min_weight:.1%}, re-optimizing on {len(kept)}.")
                    tickers = kept
                    if self.strategy == "capm":
                        expected_returns = self._calculate_capm_returns(tickers, daily_rf)
                    weights = self._optimize(tickers, expected_returns, daily_rf)
            elif n_dropped == len(tickers):
                print(f"  Warning: min_weight {self.min_weight} would drop all assets. Skipping filter.")

        # Recompute betas on final ticker set
        betas = self._calculate_betas(tickers)

        # Calculate portfolio metrics
        metrics = self._calculate_metrics(weights, tickers, expected_returns)

        # Benchmark metrics
        bench_metrics = self._calculate_benchmark_metrics()

        # Store results
        self.results = {
            "tickers": tickers,
            "weights": weights,
            "betas": betas,
            "metrics": metrics,
            "bench_metrics": bench_metrics,
            "expected_returns": expected_returns,
            "period_start": self.returns_df.index.min(),
            "period_end": self.returns_df.index.max(),
        }

        print("Optimization complete.")

    def _analyze_correlations(self, tickers):
        """Print highly correlated pairs."""
        if len(tickers) < 2:
            return

        corr = self.returns_df[tickers].corr()
        high_pairs = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                c = corr.iloc[i, j]
                if abs(c) > self.correlation_threshold:
                    high_pairs.append((tickers[i], tickers[j], c))

        if high_pairs:
            print(f"\nHighly correlated pairs (>{self.correlation_threshold:.0%}):")
            for a1, a2, c in high_pairs:
                print(f"  {a1} <-> {a2}: {c:.3f}")

    def _calculate_betas(self, tickers):
        """Calculate historical beta vs benchmark."""
        betas = {}
        bench_var = self.benchmark_returns.var()
        if bench_var < 1e-10:
            return {t: 0.0 for t in tickers}

        for t in tickers:
            try:
                cov = np.cov(self.returns_df[t], self.benchmark_returns)[0, 1]
                betas[t] = cov / bench_var
            except Exception:
                betas[t] = 0.0
        return betas

    def _calculate_capm_returns(self, tickers, daily_rf):
        """Calculate CAPM expected returns: E(r) = rf + beta * (rm - rf)."""
        bench_mean = self.benchmark_returns.mean()
        market_premium = bench_mean - daily_rf
        bench_var = self.benchmark_returns.var()

        if bench_var < 1e-10:
            return None

        expected = {}
        for t in tickers:
            try:
                cov = np.cov(self.returns_df[t], self.benchmark_returns)[0, 1]
                beta = cov / bench_var
                expected[t] = daily_rf + beta * market_premium
            except Exception:
                pass
        return expected if expected else None

    def _robust_covariance(self, tickers):
        """Compute covariance matrix with Ledoit-Wolf shrinkage if available."""
        data = self.returns_df[tickers]
        try:
            from sklearn.covariance import LedoitWolf
            return LedoitWolf().fit(data).covariance_
        except ImportError:
            return data.cov().values

    def _ensure_positive_definite(self, matrix):
        """Regularize matrix if not positive definite."""
        eigenvals = np.linalg.eigvals(matrix)
        min_eig = np.min(eigenvals)
        if min_eig < 0:
            matrix += (abs(min_eig) + 1e-6) * np.eye(len(matrix))
        return matrix

    def _get_bounds(self, n_assets):
        """Weight bounds, with fallback if max_weight too restrictive."""
        bounds = tuple((0.0, self.max_weight) for _ in range(n_assets))
        if self.max_weight * n_assets < 1.0:
            print(f"  Warning: max_weight {self.max_weight} too low for {n_assets} assets. "
                  f"Using 1.0 fallback.")
            bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        return bounds

    def _optimize(self, tickers, expected_returns, daily_rf):
        """Run the configured strategy on the given ticker subset."""
        n_assets = len(tickers)
        sigma = self._robust_covariance(tickers)
        sigma = self._ensure_positive_definite(sigma)

        if self.strategy == "minvol":
            return self._optimize_min_variance(sigma, n_assets)
        if self.strategy == "riskparity":
            return self._optimize_risk_parity(sigma, n_assets)
        mu = np.array([expected_returns[t] for t in tickers])
        return self._optimize_max_sharpe(mu, sigma, daily_rf, n_assets)

    def _optimize_min_variance(self, sigma, n_assets):
        """Minimize portfolio variance: w'Sigma*w."""
        def portfolio_variance(w):
            return w @ sigma @ w

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = self._get_bounds(n_assets)
        w0 = np.array([1 / n_assets] * n_assets)

        try:
            result = minimize(
                portfolio_variance, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if result.success:
                return result.x
        except Exception as e:
            print(f"  Optimization failed: {e}")

        print("  Fallback: Equal weights")
        return w0

    def _optimize_risk_parity(self, sigma, n_assets):
        """Risk Parity via convex formulation (Spinu 2013).

        Solves: min 0.5 * y'Σy - (1/N) * Σ ln(y_i)
        then normalizes w = y / sum(y). This is convex, always converges,
        and the log-barrier naturally pushes weights away from zero.
        """
        budget = 1.0 / n_assets

        def objective(y):
            return 0.5 * y @ sigma @ y - budget * np.sum(np.log(y))

        def gradient(y):
            return sigma @ y - budget / y

        y0 = np.ones(n_assets) / n_assets
        bounds = tuple((1e-8, None) for _ in range(n_assets))

        try:
            result = minimize(
                objective, y0, jac=gradient, method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 1000},
            )
            if result.success:
                w = result.x / result.x.sum()
                return w
        except Exception as e:
            print(f"  Optimization failed: {e}")

        print("  Fallback: Equal weights")
        return np.array([1 / n_assets] * n_assets)

    def _optimize_max_sharpe(self, mu, sigma, daily_rf, n_assets):
        """Maximize Sharpe ratio: (w'mu - rf) / sqrt(w'Sigma*w)."""
        def neg_sharpe(w):
            port_return = w @ mu
            port_var = w @ sigma @ w
            if port_var <= 1e-15:
                return 1e10
            return -(port_return - daily_rf) / np.sqrt(port_var)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = self._get_bounds(n_assets)
        w0 = np.array([1 / n_assets] * n_assets)

        try:
            result = minimize(
                neg_sharpe, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if result.success:
                return result.x
        except Exception as e:
            print(f"  Optimization failed: {e}")

        print("  Fallback: Equal weights")
        return w0

    def _calculate_metrics(self, weights, tickers, expected_returns=None):
        """Calculate portfolio performance metrics."""
        data = self.returns_df[tickers]
        sigma = data.cov().values

        # Ex-ante volatility
        ex_ante_var = weights @ sigma @ weights
        ex_ante_vol = np.sqrt(ex_ante_var) * np.sqrt(self.trading_days)

        # Historical backtest
        portfolio_returns = (data * weights).sum(axis=1)
        cum_growth = (1 + portfolio_returns).cumprod().iloc[-1]
        years = len(portfolio_returns) / self.trading_days
        hist_cagr = (cum_growth ** (1 / years)) - 1 if years > 0 else 0
        hist_vol = portfolio_returns.std() * np.sqrt(self.trading_days)
        hist_sharpe = (hist_cagr - self.annual_rf) / hist_vol if hist_vol > 0 else 0

        # Drawdown
        cum_ret = (1 + portfolio_returns).cumprod()
        peak = cum_ret.expanding().max()
        max_drawdown = ((cum_ret - peak) / peak).min()

        # VaR/CVaR
        var_95 = portfolio_returns.quantile(0.05)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

        metrics = {
            "ex_ante_vol": ex_ante_vol,
            "hist_cagr": hist_cagr,
            "hist_vol": hist_vol,
            "hist_sharpe": hist_sharpe,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95,
        }

        # CAPM expected metrics
        if expected_returns and self.strategy == "capm":
            mu_arr = np.array([expected_returns[t] for t in tickers])
            exp_ret_daily = weights @ mu_arr
            exp_annual_ret = (1 + exp_ret_daily) ** self.trading_days - 1
            exp_sharpe = (exp_annual_ret - self.annual_rf) / ex_ante_vol if ex_ante_vol > 0 else 0
            metrics["exp_return"] = exp_annual_ret
            metrics["exp_vol"] = ex_ante_vol
            metrics["exp_sharpe"] = exp_sharpe

        return metrics

    def _calculate_benchmark_metrics(self):
        """Calculate benchmark performance metrics."""
        cum = (1 + self.benchmark_returns).cumprod()
        cum_growth = cum.iloc[-1]
        years = len(self.benchmark_returns) / self.trading_days
        cagr = (cum_growth ** (1 / years)) - 1 if years > 0 else 0
        vol = self.benchmark_returns.std() * np.sqrt(self.trading_days)
        sharpe = (cagr - self.annual_rf) / vol if vol > 0 else 0
        peak = cum.expanding().max()
        max_drawdown = ((cum - peak) / peak).min()
        var_95 = self.benchmark_returns.quantile(0.05)
        cvar_95 = self.benchmark_returns[self.benchmark_returns <= var_95].mean()
        return {"cagr": cagr, "vol": vol, "sharpe": sharpe,
                "max_drawdown": max_drawdown, "var_95": var_95, "cvar_95": cvar_95}

    def generate_report(self, filename="portfolio_result.txt"):
        """Generate portfolio report as .txt and .csv."""
        if not self.results:
            print("No results. Run run_optimizer() first.")
            return

        tickers = self.results["tickers"]
        weights = self.results["weights"]
        betas = self.results["betas"]
        metrics = self.results["metrics"]
        bench = self.results["bench_metrics"]
        expected_returns = self.results["expected_returns"]

        strategy_labels = {"minvol": "MINIMUM VARIANCE", "capm": "CAPM", "riskparity": "RISK PARITY"}
        strategy_label = strategy_labels[self.strategy]
        bench_name = self.benchmark_ticker

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{strategy_label} PORTFOLIO RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Rapport skapad: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Period: {self.results['period_start'].strftime('%Y-%m-%d')} – {self.results['period_end'].strftime('%Y-%m-%d')}\n")
            f.write(f"Strategy: {strategy_label}\n")
            f.write(f"Benchmark: {bench_name}\n")
            if self.strategy == "capm":
                f.write(f"Risk-Free Rate: {self.annual_rf:.2%}\n")
            f.write(f"Max weight per asset: {self.max_weight:.1%}\n")
            if self.sector_filter:
                f.write(f"Sector filter: {self.sector_filter}\n")
            f.write("\n")

            # Asset allocations
            total_inv = 100_000
            if self.strategy == "capm" and expected_returns:
                f.write(f"{'Asset':<15} {'Weight':>10} {'Beta':>8} {'Exp.Ret':>12} "
                        f"{'Alloc (100k)':>15}\n")
                f.write("-" * 65 + "\n")
            else:
                f.write(f"{'Asset':<15} {'Weight':>10} {'Beta':>8} {'Alloc (100k)':>15}\n")
                f.write("-" * 55 + "\n")

            for i, t in enumerate(tickers):
                w = weights[i]
                if w < 0.001:
                    continue
                b = betas.get(t, 0)
                if self.strategy == "capm" and expected_returns:
                    er = (1 + expected_returns.get(t, 0)) ** self.trading_days - 1
                    f.write(f"{t:<15} {w:>10.1%} {b:>8.2f} {er:>12.1%} "
                            f"{w * total_inv:>14,.0f} SEK\n")
                else:
                    f.write(f"{t:<15} {w:>10.1%} {b:>8.2f} "
                            f"{w * total_inv:>14,.0f} SEK\n")

            f.write("\n")

            # Performance comparison
            bench_col = f"Benchmark ({bench_name})"

            if "exp_return" in metrics:
                f.write(f"{'Metric':<20} {'Expected (CAPM)':>18} {'Historical':>18} "
                        f"{bench_col:>22}\n")
                f.write("-" * 82 + "\n")
                f.write(f"{'CAGR':<20} {metrics['exp_return']:>18.2%} "
                        f"{metrics['hist_cagr']:>18.2%} {bench['cagr']:>22.2%}\n")
                f.write(f"{'Volatility':<20} {metrics['exp_vol']:>18.2%} "
                        f"{metrics['hist_vol']:>18.2%} {bench['vol']:>22.2%}\n")
                f.write(f"{'Sharpe Ratio':<20} {metrics['exp_sharpe']:>18.2f} "
                        f"{metrics['hist_sharpe']:>18.2f} {bench['sharpe']:>22.2f}\n")
            else:
                f.write(f"{'Metric':<20} {'Historical':>18} {bench_col:>22}\n")
                f.write("-" * 65 + "\n")
                f.write(f"{'CAGR':<20} {metrics['hist_cagr']:>18.2%} "
                        f"{bench['cagr']:>22.2%}\n")
                f.write(f"{'Volatility':<20} {metrics['hist_vol']:>18.2%} "
                        f"{bench['vol']:>22.2%}\n")
                f.write(f"{'(Ex-Ante Vol)':<20} {metrics['ex_ante_vol']:>18.2%} "
                        f"{'---':>22}\n")
                f.write(f"{'Sharpe Ratio':<20} {metrics['hist_sharpe']:>18.2f} "
                        f"{bench['sharpe']:>22.2f}\n")

            f.write("\n")

            # Risk metrics
            bench_col = f"Benchmark ({bench_name})"
            f.write(f"{'Risk Metrics':<25} {'Value':>10} {bench_col:>22}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Max Drawdown':<25} {metrics['max_drawdown']:>10.2%} {bench['max_drawdown']:>22.2%}\n")
            f.write(f"{'Daily VaR (95%)':<25} {metrics['var_95']:>10.2%} {bench['var_95']:>22.2%}\n")
            f.write(f"{'Daily CVaR (95%)':<25} {metrics['cvar_95']:>10.2%} {bench['cvar_95']:>22.2%}\n")

            f.write("\n")

            # Yearly returns (portfolio vs benchmark)
            port_r = (self.returns_df[tickers] * weights).sum(axis=1)
            port_yearly = (1 + port_r).groupby(port_r.index.year).prod() - 1
            bench_yearly = (1 + self.benchmark_returns).groupby(
                self.benchmark_returns.index.year).prod() - 1

            f.write(f"{'Year':<10} {'Portfolio':>12} {bench_name:>22}\n")
            f.write("-" * 48 + "\n")
            for year in port_yearly.index:
                pr = port_yearly.loc[year]
                br = bench_yearly.loc[year] if year in bench_yearly.index else float("nan")
                br_str = f"{br:>22.2%}" if pd.notna(br) else f"{'---':>22}"
                f.write(f"{year:<10} {pr:>12.2%} {br_str}\n")

            f.write("\n")
            if self.strategy == "minvol":
                f.write("NOTE: MinVol minimizes historical volatility, ignoring return predictions.\n")
                f.write("'Historical' metrics show how this portfolio would have performed.\n")
            elif self.strategy == "riskparity":
                f.write("NOTE: Risk Parity equalizes each asset's contribution to total portfolio risk.\n")
                f.write("'Historical' metrics show how this portfolio would have performed.\n")
            else:
                f.write("NOTE: 'Historical' = in-sample backtest on training data.\n")
                f.write("'Expected' = theoretical CAPM projections (use with caution).\n")

        print(f"Report saved to {filename}")

        # Save CSV
        csv_filename = filename.replace(".txt", ".csv")
        rows = []
        for i, t in enumerate(tickers):
            w = weights[i]
            if w < 0.001:
                continue
            row = {
                "Ticker": t,
                "Sector": self.sector_map.get(t, "Unknown"),
                "Weight": w,
                "Beta": betas.get(t, 0),
                "Alloc_100k": w * 100_000,
            }
            if expected_returns and self.strategy == "capm":
                row["Exp_Return"] = (1 + expected_returns.get(t, 0)) ** self.trading_days - 1
            rows.append(row)

        csv_df = pd.DataFrame(rows)
        csv_df.to_csv(csv_filename, index=False)
        print(f"CSV saved to {csv_filename}")

        # PNG
        png_filename = filename.replace(".txt", ".png")
        self.generate_charts(png_filename)

    def generate_charts(self, filename="portfolio_result.png"):
        """Save portfolio visualization as PNG (4 panels)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available, skipping charts.")
            return

        if not self.results:
            return

        tickers = self.results["tickers"]
        weights = self.results["weights"]
        metrics = self.results["metrics"]

        # Active positions sorted by weight descending
        pairs = sorted(
            [(t, w, i) for i, (t, w) in enumerate(zip(tickers, weights)) if w >= 0.001],
            key=lambda x: -x[1],
        )
        tickers_s = [p[0] for p in pairs]
        weights_s = np.array([p[1] for p in pairs])
        active_idx = [p[2] for p in pairs]

        # Cumulative returns
        port_r = (self.returns_df[tickers] * weights).sum(axis=1)
        cum_port = (1 + port_r).cumprod()
        cum_bench = (1 + self.benchmark_returns).cumprod()

        # Drawdown
        peak = cum_port.expanding().max()
        dd = (cum_port - peak) / peak

        # Risk contributions (fraction of total variance)
        sigma = self.returns_df[tickers].cov().values
        port_var = float(weights @ sigma @ weights)
        rc_full = (weights * (sigma @ weights)) / port_var if port_var > 1e-15 else weights.copy()
        rc_s = np.array([rc_full[i] for i in active_idx])

        # Sector → color mapping
        unique_sectors = sorted(set(self.sector_map.get(t, "Unknown") for t in tickers_s))
        tab10 = plt.cm.tab10
        sec_color = {s: tab10(i % 10) for i, s in enumerate(unique_sectors)}

        # ── Layout ──────────────────────────────────────────────────────────
        bg = "#0f1117"
        grid_c = "#1e2030"

        fig = plt.figure(figsize=(16, 13))
        fig.patch.set_facecolor(bg)

        strategy_labels = {
            "minvol": "Minimum Variance",
            "capm": "CAPM (Max Sharpe)",
            "riskparity": "Risk Parity",
        }
        fig.suptitle(
            f"{strategy_labels[self.strategy]}  ·  "
            f"CAGR {metrics['hist_cagr']:.1%}  ·  "
            f"Sharpe {metrics['hist_sharpe']:.2f}  ·  "
            f"Max DD {metrics['max_drawdown']:.1%}",
            fontsize=12, color="white", y=0.99, fontweight="bold",
        )

        gs = gridspec.GridSpec(
            3, 2, figure=fig,
            height_ratios=[2, 1.2, 1.5],
            hspace=0.45, wspace=0.08,
            left=0.07, right=0.97, top=0.95, bottom=0.10,
        )
        ax_cum = fig.add_subplot(gs[0, :])
        ax_dd  = fig.add_subplot(gs[1, :])
        ax_wt  = fig.add_subplot(gs[2, 0])
        ax_rc  = fig.add_subplot(gs[2, 1], sharey=ax_wt)

        def _style(ax):
            ax.set_facecolor(bg)
            ax.tick_params(colors="#888888", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(grid_c)
            ax.grid(True, color=grid_c, lw=0.5)

        for ax in [ax_cum, ax_dd, ax_wt, ax_rc]:
            _style(ax)

        # ── Panel 1: Kumulativ avkastning ────────────────────────────────────
        ax_cum.plot(cum_port.index, cum_port.values, color="#4fc3f7", lw=1.5, label="Portfolio")
        ax_cum.plot(cum_bench.index, cum_bench.values, color="#ef5350", lw=1.2,
                    linestyle="--", alpha=0.7, label=self.benchmark_ticker)
        period_str = (f"{self.results['period_start'].strftime('%Y-%m-%d')} – "
                      f"{self.results['period_end'].strftime('%Y-%m-%d')}")
        ax_cum.set_title(f"Kumulativ avkastning  ·  {period_str}", color="white", fontsize=10, pad=5)
        ax_cum.legend(fontsize=8, facecolor=bg, labelcolor="white",
                      framealpha=0.4, edgecolor=grid_c)
        ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
        ax_cum.set_xlim(cum_port.index[0], cum_port.index[-1])

        # ── Panel 2: Drawdown ────────────────────────────────────────────────
        ax_dd.fill_between(dd.index, dd.values, 0, color="#ef5350", alpha=0.35)
        ax_dd.plot(dd.index, dd.values, color="#ef5350", lw=0.9)
        ax_dd.set_title("Drawdown", color="white", fontsize=10, pad=5)
        ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_dd.set_xlim(dd.index[0], dd.index[-1])

        # ── Panel 3: Vikter ──────────────────────────────────────────────────
        n = len(tickers_s)
        y_pos = np.arange(n)
        bar_colors = [sec_color[self.sector_map.get(t, "Unknown")] for t in tickers_s]

        bars_wt = ax_wt.barh(y_pos, weights_s * 100, color=bar_colors,
                              edgecolor="none", height=0.7)
        ax_wt.set_yticks(y_pos)
        ax_wt.set_yticklabels(tickers_s, fontsize=7, color="white")
        ax_wt.invert_yaxis()  # Största vikt överst
        ax_wt.set_xlabel("Vikt (%)", color="#888888", fontsize=8)
        ax_wt.set_title("Vikter", color="white", fontsize=10, pad=5)
        ax_wt.grid(True, axis="x", color=grid_c, lw=0.5)
        ax_wt.grid(False, axis="y")
        for bar, w in zip(bars_wt, weights_s):
            ax_wt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                       f"{w:.1%}", va="center", fontsize=7, color="white")
        # Max-weight-gräns som vertikal linje
        max_w_pct = self.max_weight * 100
        if max_w_pct < 100:
            ax_wt.axvline(max_w_pct, color="#ffa726", linestyle=":", lw=1, alpha=0.8)
            ax_wt.text(max_w_pct, -0.8, f"  max {self.max_weight:.0%}",
                       color="#ffa726", fontsize=6.5, va="bottom", ha="left")

        # Sektorlegend placeras under de två nedre panelerna
        handles = [plt.Rectangle((0, 0), 1, 1, color=sec_color[s]) for s in unique_sectors]
        fig.legend(handles, unique_sectors, fontsize=6.5, facecolor=bg,
                   labelcolor="white", framealpha=0.4, edgecolor=grid_c,
                   loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=min(len(unique_sectors), 6))

        # ── Panel 4: Riskbidrag (delar y-axel med Vikter) ─────────────────────
        ax_rc.barh(y_pos, rc_s * 100, color=bar_colors, edgecolor="none", height=0.7)
        plt.setp(ax_rc.get_yticklabels(), visible=False)
        ax_rc.set_xlabel("Andel av varians (%)", color="#888888", fontsize=8)
        ax_rc.set_title("Riskbidrag", color="white", fontsize=10, pad=5)
        ax_rc.grid(True, axis="x", color=grid_c, lw=0.5)
        ax_rc.grid(False, axis="y")
        for patch, rv in zip(ax_rc.patches, rc_s):
            ax_rc.text(patch.get_width() + 0.3, patch.get_y() + patch.get_height() / 2,
                       f"{rv:.1%}", va="center", fontsize=7, color="white")

        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=bg)
        plt.close()
        print(f"Chart saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Optimizer (MinVol or CAPM strategy)"
    )
    parser.add_argument(
        "--strategy", choices=["minvol", "capm", "riskparity"], default="minvol",
        help="Optimization strategy: minvol (default), capm, or riskparity",
    )
    parser.add_argument(
        "--portfolio", type=str, default="portfolio.csv",
        help="Portfolio file (default: portfolio.csv)",
    )
    parser.add_argument(
        "--max-weight", type=float, default=None,
        help="Max weight per asset (default: 0.15 for minvol, 0.60 for capm, 1.0 for riskparity)",
    )
    parser.add_argument(
        "--min-weight", type=float, default=0.0,
        help="Drop positions below this threshold post-optimization and renormalize (default: 0.0)",
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Years of history to analyze (default: 5)",
    )
    parser.add_argument(
        "--sector", type=str, default=None,
        help="Filter by sector (e.g., 'Industrials')",
    )
    parser.add_argument(
        "--rf", type=float, default=0.02,
        help="Annual risk-free rate (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--benchmark", type=str, default="^OMXSBCAPGI",
        help="Benchmark ticker (default: ^OMXSBCAPGI)",
    )

    args = parser.parse_args()

    max_weight = args.max_weight
    if max_weight is None:
        max_weight = {"minvol": 0.15, "capm": 0.60, "riskparity": 1.0}[args.strategy]

    output_file = f"portfolio_{args.strategy}.txt"

    optimizer = PortfolioOptimizer(
        data_dir="data",
        portfolio_file=args.portfolio,
        strategy=args.strategy,
        annual_rf=args.rf,
        max_weight=max_weight,
        min_weight=args.min_weight,
        max_years=args.years,
        sector_filter=args.sector,
        benchmark_ticker=args.benchmark,
    )

    optimizer.sector_map = utils.load_sectors("sectors.csv")
    optimizer.load_data()

    if optimizer.prices_df is None:
        print("Failed to load data. Exiting.")
        return

    optimizer.run_optimizer()
    optimizer.generate_report(output_file)


if __name__ == "__main__":
    main()
