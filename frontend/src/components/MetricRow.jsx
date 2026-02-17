import InfoTooltip from "./InfoTooltip";

export default function MetricRow({ label, value, tooltip }) {
  return (
    <div className="metric-row">
      <div className="metric-row__label">
        {label}
        {tooltip && <InfoTooltip text={tooltip} />}
      </div>
      <div className="metric-row__value">{value ?? "—"}</div>
    </div>
  );
}
