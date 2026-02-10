export default function Portfolio({ holdings }) {
  if (holdings.length === 0) {
    return <div className="panel-block">No holdings added yet.</div>;
  }

  return (
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Shares</th>
          <th>Bought At</th>
          <th>Price</th>
          <th>Weight</th>
        </tr>
      </thead>
      <tbody>
        {holdings.map((h) => (
          <tr key={h.id}>
            <td>{h.ticker}</td>
            <td>{h.shares}</td>
            <td>{h.asOfDate}</td>
            <td>{h.buyPrice ?? "—"}</td>
            <td>{h.weight ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
