import AnalyticsCard from "./AnalyticsCard";
import MetricRow from "./MetricRow";
import RegimeStatesTable from "./RegimeStatesTable";
import { formatPct, formatNum, formatMoney } from "../lib/formatters";

function RiskMetricsBlock({ metrics }) {
  if (!metrics) return null;

  return (
    <>
      <MetricRow
        label="Annualized Return"
        value={formatPct(metrics.annualized_return)}
        tooltip="The geometric average return per year, accounting for compounding."
      />
      <MetricRow
        label="Annualized Volatility"
        value={formatPct(metrics.annualized_volatility)}
        tooltip="The standard deviation of daily returns, annualized."
      />
      <MetricRow
        label="Max Drawdown"
        value={formatPct(metrics.max_drawdown)}
        tooltip="The maximum observed loss from a peak to a trough of the portfolio, before a new peak is attained."
      />
      <MetricRow
        label="Sharpe Ratio"
        value={formatNum(metrics.sharpe_ratio, 2)}
        tooltip="Annualized return divided by annualized volatility. Higher = better risk-adjusted return."
      />
    </>
  );
}

function ForecastSummaryBlock({ summary }) {
  if (!summary) return null;

  return (
    <>
      <MetricRow
        label="Last Historical Value"
        value={formatMoney(summary.last_historical_value)}
        tooltip="Value of the portfolio at the end of the historical period (i.e. the last actual value before the forecast starts)."
      />
      <MetricRow
        label="Forecast End Value"
        value={formatMoney(summary.forecast_end_value)}
        tooltip="The projected value of the portfolio at the end of the forecast period."
      />
      <MetricRow
        label="Forecast Abs Change"
        value={formatMoney(summary.forecast_abs_change)}
        tooltip="The absolute change in portfolio value over the forecast period (end value − last historical value)."
      />
      <MetricRow
        label="Forecast Total Return"
        value={formatPct(summary.forecast_total_return)}
        tooltip="The total return over the forecast period, calculated as (end value / last historical value) − 1."
      />
      <MetricRow
        label="Avg Daily Return (Forecast)"
        value={formatPct(summary.forecast_avg_daily_return)}
        tooltip="The average daily return over the forecast period."
      />
      <MetricRow
        label={`Days to ${((summary.target_multiple ?? 1.1) * 100).toFixed(
          0,
        )}% Target`}
        value={
          summary.days_to_target_multiple == null
            ? "—"
            : `${summary.days_to_target_multiple} days`
        }
        tooltip="The number of days it takes for the portfolio to reach a target multiple of its last historical value."
      />
    </>
  );
}

function StochasticForecastSummaryBlock({
  terminal,
  drawdown,
  volatility,
  variance,
}) {
  if (!terminal && !drawdown) return null;

  const fmtDrawdown = (x) => (x == null ? "—" : `${(-x * 100).toFixed(2)}%`);

  // variance is per-path terminal variance (daily units) -- convert to
  // annualized volatility so it reads in the same units as every other
  // volatility figure in the app, rather than a raw, hard-to-interpret
  // variance number.
  const annualizedVolFromVariance = (v) =>
    v == null ? undefined : Math.sqrt(v * 252);

  return (
    <>
      <MetricRow
        label="Mean Terminal Value"
        value={formatMoney(terminal?.mean_terminal_value)}
        tooltip="The average ending portfolio value across all Monte Carlo simulations."
      />
      <MetricRow
        label="Median Terminal Value"
        value={formatMoney(terminal?.median_terminal_value)}
        tooltip="The median ending portfolio value across all Monte Carlo simulations."
      />
      <MetricRow
        label="Bear Case"
        value={formatMoney(terminal?.bear_case)}
        tooltip="A downside scenario based on the lower tail of simulated outcomes."
      />
      <MetricRow
        label="Bull Case"
        value={formatMoney(terminal?.bull_case)}
        tooltip="An upside scenario based on the upper tail of simulated outcomes."
      />
      <MetricRow
        label="Probability of Loss"
        value={formatPct(terminal?.probability_of_loss)}
        tooltip="The fraction of simulations where the portfolio ends below its starting forecast value."
      />
      <MetricRow
        label="Median Max Drawdown"
        value={fmtDrawdown(drawdown?.median_max_drawdown)}
        tooltip="The typical worst peak-to-trough decline across simulated paths."
      />
      <MetricRow
        label="Prob. Drawdown > 20%"
        value={formatPct(drawdown?.prob_drawdown_gt_20)}
        tooltip="The fraction of simulations where the portfolio experiences a drawdown of at least 20%."
      />
      <MetricRow
        label="Annualized Volatility"
        value={formatPct(volatility?.annualized_volatility)}
        tooltip="The forecast volatility parameter used in the stochastic simulation, annualized. Only set for GBM, which uses one fixed volatility for the whole simulation."
      />
      {variance && (
        <>
          <MetricRow
            label="Mean Terminal Volatility"
            value={formatPct(
              annualizedVolFromVariance(variance.mean_terminal_variance),
            )}
            tooltip="Annualized volatility implied by the average terminal variance across simulated paths. Set for models with path-dependent variance (GARCH, Heston, regime-switching), not GBM."
          />
          <MetricRow
            label="Median Terminal Volatility"
            value={formatPct(
              annualizedVolFromVariance(variance.median_terminal_variance),
            )}
            tooltip="Annualized volatility implied by the median terminal variance across simulated paths."
          />
          <MetricRow
            label="P10 Terminal Volatility"
            value={formatPct(
              annualizedVolFromVariance(variance.p10_terminal_variance),
            )}
            tooltip="Annualized volatility implied by the 10th percentile terminal variance across simulated paths -- the calmer end of the distribution."
          />
          <MetricRow
            label="P90 Terminal Volatility"
            value={formatPct(
              annualizedVolFromVariance(variance.p90_terminal_variance),
            )}
            tooltip="Annualized volatility implied by the 90th percentile terminal variance across simulated paths -- the stormier end of the distribution."
          />
        </>
      )}
    </>
  );
}

function getForecastType(fc) {
  return fc?.inputs?.forecast?.type || "deterministic";
}

function getShockType(sr) {
  return sr?.inputs?.shock?.type;
}

function shockTitle(shockType) {
  if (shockType === "calibrated_regime") {
    return "Calibrated Regime-Switching Stress Test";
  }
  if (shockType === "deterministic_regime") {
    return "Deterministic Regime-Shift Stress Test";
  }
  return "Stress Test";
}

export default function AnalyticsPanel({
  analysis,
  forecast,
  stressResult,
  regimePreview,
  selectedState,
  onSelectState,
}) {
  const hasAnalysis = !!analysis?.metrics;

  const forecastType = getForecastType(forecast);
  const hasDeterministicForecast =
    forecastType === "deterministic" && !!forecast?.summary;
  const hasStochasticForecast =
    forecastType === "stochastic" &&
    (!!forecast?.terminal || !!forecast?.drawdown);

  const hasRegimePreview = !!regimePreview?.regime_stats?.length;
  const hasStressForecast = !!stressResult?.terminal || !!stressResult?.drawdown;

  return (
    <div className="analytics-panel">
      {hasAnalysis && (
        <AnalyticsCard title="Risk Metrics">
          <RiskMetricsBlock metrics={analysis.metrics} />
        </AnalyticsCard>
      )}

      {hasDeterministicForecast && (
        <AnalyticsCard title="Forecast Summary">
          <ForecastSummaryBlock summary={forecast.summary} />
        </AnalyticsCard>
      )}

      {hasStochasticForecast && (
        <AnalyticsCard title="Stochastic Forecast Summary">
          <StochasticForecastSummaryBlock
            terminal={forecast.terminal}
            drawdown={forecast.drawdown}
            volatility={forecast.volatility}
            variance={forecast.variance}
          />
        </AnalyticsCard>
      )}

      {/* calibrated_regime step 1 -- fitted states, pick one to stress test */}
      {hasRegimePreview && (
        <AnalyticsCard title="Regime States (fitted)">
          <RegimeStatesTable
            regimeStats={regimePreview.regime_stats}
            selectedState={selectedState}
            onSelectState={onSelectState}
          />
        </AnalyticsCard>
      )}

      {/* Either shock mechanism -- same response shape, same block. No
          "volatility" prop: that's a single GBM input parameter, never
          present on a stress response. "variance" IS present for
          calibrated_regime (the regime-switching process has genuinely
          path-dependent variance) and renders as annualized volatility
          rows; deterministic_regime has none (constant sigma), so those
          rows just don't appear -- same StochasticForecastSummaryBlock
          handles both correctly since it's null-guarded already. */}
      {hasStressForecast && (
        <AnalyticsCard title={shockTitle(getShockType(stressResult))}>
          <StochasticForecastSummaryBlock
            terminal={stressResult.terminal}
            drawdown={stressResult.drawdown}
            variance={stressResult.variance}
          />
        </AnalyticsCard>
      )}
    </div>
  );
}
