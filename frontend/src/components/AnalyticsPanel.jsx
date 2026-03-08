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
        tooltip="The number of days it takes for the portfolio to reach a target multiple of its last historical value."
      />
    </>
  );
}

function StochasticForecastSummaryBlock({ terminal, drawdown, volatility }) {
  if (!terminal && !drawdown) return null;

  const fmtDrawdown = (x) => (x == null ? "—" : `${(-x * 100).toFixed(2)}%`);

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
        tooltip="The forecast volatility parameter used in the stochastic simulation, annualized."
      />
    </>
  );
}

function getForecastType(fc) {
  return fc?.inputs?.forecast?.type || "deterministic";
}

export default function AnalyticsPanel({
  analysis,
  stress,
  forecast,
  stressForecast,
}) {
  const hasAnalysis = !!analysis?.metrics;

  const hasStress = !!stress?.baseline?.metrics && !!stress?.scenario?.metrics;
  const hasStressDelta = !!stress?.delta?.metrics;

  const forecastType = getForecastType(forecast);
  const stressBaselineForecastType = getForecastType(stressForecast?.baseline);
  const stressScenarioForecastType = getForecastType(stressForecast?.scenario);

  const hasDeterministicForecast =
    forecastType === "deterministic" && !!forecast?.summary;

  const hasStochasticForecast =
    forecastType === "stochastic" &&
    (!!forecast?.terminal || !!forecast?.drawdown);

  const hasStressDeterministicForecast =
    stressBaselineForecastType === "deterministic" &&
    stressScenarioForecastType === "deterministic" &&
    !!stressForecast?.baseline?.summary &&
    !!stressForecast?.scenario?.summary;

  const hasStressStochasticForecast =
    stressBaselineForecastType === "stochastic" &&
    stressScenarioForecastType === "stochastic" &&
    !!stressForecast?.baseline?.terminal &&
    !!stressForecast?.scenario?.terminal;

  const deterministicForecastDelta = hasStressDeterministicForecast
    ? diffObjects(
        stressForecast.baseline.summary,
        stressForecast.scenario.summary,
      )
    : null;

  const stochasticTerminalDelta = hasStressStochasticForecast
    ? diffObjects(
        stressForecast.baseline.terminal,
        stressForecast.scenario.terminal,
      )
    : null;

  const stochasticDrawdownDelta = hasStressStochasticForecast
    ? diffObjects(
        stressForecast.baseline.drawdown,
        stressForecast.scenario.drawdown,
      )
    : null;

  const stochasticVolatilityDelta = hasStressStochasticForecast
    ? diffObjects(
        stressForecast.baseline.volatility,
        stressForecast.scenario.volatility,
      )
    : null;

  const isStressForecastMode =
    hasStress &&
    (hasStressDeterministicForecast || hasStressStochasticForecast);

  return (
    <div className="analytics-panel">
      {/* ---- Non-stress (plain) ---- */}
      {hasAnalysis && !hasStress && (
        <AnalyticsCard title="Risk Metrics">
          <RiskMetricsBlock metrics={analysis.metrics} />
        </AnalyticsCard>
      )}

      {hasDeterministicForecast && !hasStressDeterministicForecast && (
        <AnalyticsCard title="Forecast Summary">
          <ForecastSummaryBlock summary={forecast.summary} />
        </AnalyticsCard>
      )}

      {hasStochasticForecast && !hasStressStochasticForecast && (
        <AnalyticsCard title="Stochastic Forecast Summary">
          <StochasticForecastSummaryBlock
            terminal={forecast.terminal}
            drawdown={forecast.drawdown}
            volatility={forecast.volatility}
          />
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
            {hasStressDeterministicForecast ? (
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
                  <ForecastSummaryBlock summary={deterministicForecastDelta} />
                </AnalyticsCard>
              </>
            ) : (
              <>
                <AnalyticsCard title="Stochastic Forecast Summary (Baseline)">
                  <StochasticForecastSummaryBlock
                    terminal={stressForecast.baseline.terminal}
                    drawdown={stressForecast.baseline.drawdown}
                    volatility={stressForecast.baseline.volatility}
                  />
                </AnalyticsCard>

                <AnalyticsCard title="Stochastic Forecast Summary (Scenario)">
                  <StochasticForecastSummaryBlock
                    terminal={stressForecast.scenario.terminal}
                    drawdown={stressForecast.scenario.drawdown}
                    volatility={stressForecast.scenario.volatility}
                  />
                </AnalyticsCard>

                <AnalyticsCard title="Stochastic Forecast Δ (Scenario − Baseline)">
                  <StochasticForecastSummaryBlock
                    terminal={stochasticTerminalDelta}
                    drawdown={stochasticDrawdownDelta}
                    volatility={stochasticVolatilityDelta}
                  />
                </AnalyticsCard>
              </>
            )}
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
          {hasStressDeterministicForecast && (
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
                <ForecastSummaryBlock summary={deterministicForecastDelta} />
              </AnalyticsCard>
            </>
          )}

          {hasStressStochasticForecast && (
            <>
              <AnalyticsCard title="Stochastic Forecast Summary (Baseline)">
                <StochasticForecastSummaryBlock
                  terminal={stressForecast.baseline.terminal}
                  drawdown={stressForecast.baseline.drawdown}
                  volatility={stressForecast.baseline.volatility}
                />
              </AnalyticsCard>

              <AnalyticsCard title="Stochastic Forecast Summary (Scenario)">
                <StochasticForecastSummaryBlock
                  terminal={stressForecast.scenario.terminal}
                  drawdown={stressForecast.scenario.drawdown}
                  volatility={stressForecast.scenario.volatility}
                />
              </AnalyticsCard>

              <AnalyticsCard title="Stochastic Forecast Δ (Scenario − Baseline)">
                <StochasticForecastSummaryBlock
                  terminal={stochasticTerminalDelta}
                  drawdown={stochasticDrawdownDelta}
                  volatility={stochasticVolatilityDelta}
                />
              </AnalyticsCard>
            </>
          )}
        </>
      )}
    </div>
  );
}
