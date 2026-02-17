import { formatPrice } from "../lib/formatters";

export default function Portfolio({ holdings, onRemove }) {
  if (holdings.length === 0) {
    return (
      <div className="panel-block">
        No holdings added yet. Use the form below.
      </div>
    );
  }

  return (
    <table className="holdings-table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Shares</th>
          <th>Bought At</th>
          <th>Price</th>
          {/* <th>Weight</th> */}
          <th></th> {/* Remove holdings */}
        </tr>
      </thead>

      <tbody>
        {holdings.map((h) => (
          <tr key={h.id}>
            <td>{h.ticker}</td>
            <td>{h.shares}</td>
            <td>{h.asOfDate}</td>
            <td>{formatPrice(h.buyPrice)}</td>
            {/* <td>{h.weight ?? "—"}</td> */}

            <td>
              <button
                type="button"
                className="remove-btn"
                onClick={() => onRemove?.(h.id)}
                title="Remove holding"
                aria-label={`Remove ${h.ticker}`}
              >
                -
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
