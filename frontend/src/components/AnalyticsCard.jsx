export default function AnalyticsCard({ title, children }) {
  return (
    <div className="analytics-card">
      <div className="analytics-card__header">
        <h3 className="analytics-card__title">{title}</h3>
      </div>
      <div className="analytics-card__body">{children}</div>
    </div>
  );
}
