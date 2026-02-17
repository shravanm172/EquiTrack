import AnalyticsCard from "./AnalyticsCard";
import MetricRow from "./MetricRow";
import {
  formatPct,
  formatNum,
  formatMoney,
  diffObjects,
} from "../lib/formatters";

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
        tooltip={`The number of days it takes for the portfolio to reach a target multiple of its last historical value. The default target multiple is 1.1 (i.e. a 10% gain)`}
      />
    </>
  );
}

export default function AnalyticsPanel({
  analysis,
  stress,
  forecast,
  stressForecast,
}) {
  // ---- Determine what’s available ----
  const hasAnalysis = !!analysis?.metrics;

  const hasStress = !!stress?.baseline?.metrics && !!stress?.scenario?.metrics;
  const hasStressDelta = !!stress?.delta?.metrics;

  const hasForecast = !!forecast?.summary;

  const hasStressForecast =
    !!stressForecast?.baseline?.summary && !!stressForecast?.scenario?.summary;

  // ---- Compute forecast deltas that aren’t provided by backend ----
  const forecastDelta = hasStressForecast
    ? diffObjects(
        stressForecast.baseline.summary,
        stressForecast.scenario.summary,
      )
    : null;

  // When stress forecast exists, force 2 rows:
  // Row 1 = risk metrics cards, Row 2 = forecast cards
  const isStressForecastMode = hasStress && hasStressForecast;

  return (
    <div className="analytics-panel">
      {/* ---- Non-stress (plain) ---- */}
      {hasAnalysis && !hasStress && (
        <AnalyticsCard title="Risk Metrics">
          <RiskMetricsBlock metrics={analysis.metrics} />
        </AnalyticsCard>
      )}

      {hasForecast && !hasStressForecast && (
        <AnalyticsCard title="Forecast Summary">
          <ForecastSummaryBlock summary={forecast.summary} />
        </AnalyticsCard>
      )}

      {/* ---- Stress forecast mode (2-row layout) ---- */}
      {isStressForecastMode ? (
        <>
          <div className="analytics-row analytics-row--risk">
            <AnalyticsCard title="Risk Metrics (Baseline)">
              <RiskMetricsBlock metrics={stress.baseline.metrics} />
            </AnalyticsCard>

            <AnalyticsCard title="Risk Metrics (Scenario)">
              <RiskMetricsBlock metrics={stress.scenario.metrics} />
            </AnalyticsCard>

            {hasStressDelta && (
              <AnalyticsCard title="Risk Metrics Δ (Scenario − Baseline)">
                <RiskMetricsBlock metrics={stress.delta.metrics} />
              </AnalyticsCard>
            )}
          </div>

          <div className="analytics-row analytics-row--forecast">
            <AnalyticsCard title="Forecast Summary (Baseline)">
              <ForecastSummaryBlock summary={stressForecast.baseline.summary} />
            </AnalyticsCard>

            <AnalyticsCard title="Forecast Summary (Scenario)">
              <ForecastSummaryBlock summary={stressForecast.scenario.summary} />
            </AnalyticsCard>

            <AnalyticsCard title="Forecast Summary Δ (Scenario − Baseline)">
              <ForecastSummaryBlock summary={forecastDelta} />
            </AnalyticsCard>
          </div>
        </>
      ) : (
        <>
          {/* ---- Stress only (no forecast) ---- */}
          {hasStress && (
            <div className="analytics-row analytics-row--risk">
              <AnalyticsCard title="Risk Metrics (Baseline)">
                <RiskMetricsBlock metrics={stress.baseline.metrics} />
              </AnalyticsCard>

              <AnalyticsCard title="Risk Metrics (Scenario)">
                <RiskMetricsBlock metrics={stress.scenario.metrics} />
              </AnalyticsCard>

              {hasStressDelta && (
                <AnalyticsCard title="Risk Metrics Δ (Scenario − Baseline)">
                  <RiskMetricsBlock metrics={stress.delta.metrics} />
                </AnalyticsCard>
              )}
            </div>
          )}

          {/* ---- Stress forecast only (edge case) ---- */}
          {hasStressForecast && (
            <>
              <AnalyticsCard title="Forecast Summary (Baseline)">
                <ForecastSummaryBlock
                  summary={stressForecast.baseline.summary}
                />
              </AnalyticsCard>

              <AnalyticsCard title="Forecast Summary (Scenario)">
                <ForecastSummaryBlock
                  summary={stressForecast.scenario.summary}
                />
              </AnalyticsCard>

              <AnalyticsCard title="Forecast Summary Δ (Scenario − Baseline)">
                <ForecastSummaryBlock summary={forecastDelta} />
              </AnalyticsCard>
            </>
          )}
        </>
      )}
    </div>
  );
}
