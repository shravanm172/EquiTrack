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
      />
      <MetricRow
        label="Annualized Volatility"
        value={formatPct(metrics.annualized_volatility)}
      />
      <MetricRow label="Max Drawdown" value={formatPct(metrics.max_drawdown)} />
      <MetricRow
        label="Sharpe Ratio"
        value={formatNum(metrics.sharpe_ratio, 2)}
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
      />
      <MetricRow
        label="Forecast End Value"
        value={formatMoney(summary.forecast_end_value)}
      />
      <MetricRow
        label="Forecast Abs Change"
        value={formatMoney(summary.forecast_abs_change)}
      />
      <MetricRow
        label="Forecast Total Return"
        value={formatPct(summary.forecast_total_return)}
      />
      <MetricRow
        label="Avg Daily Return (Forecast)"
        value={formatPct(summary.forecast_avg_daily_return)}
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
  // ---- Decide what’s available ----
  const hasAnalysis = !!analysis?.metrics;

  const hasStress = !!stress?.baseline?.metrics && !!stress?.scenario?.metrics;
  const hasStressDelta = !!stress?.delta?.metrics;

  const hasForecast = !!forecast?.summary;

  const hasStressForecast =
    !!stressForecast?.baseline?.summary && !!stressForecast?.scenario?.summary;

  // ---- Compute deltas that aren’t provided by backend ----
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
