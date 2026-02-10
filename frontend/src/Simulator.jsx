import { useState } from "react";
import Portfolio from "./components/Portfolio";
import AddHoldingForm from "./components/AddHoldingForm";

export default function SimulatorPage() {
  const [holdings, setHoldings] = useState([]);
  const [error, setError] = useState("");

  return (
    <div className="simulator-page">
      <div className="simulator-grid">
        {/* Left: Portfolio / Controls */}
        <section className="simulator-panel input-panel">
          <h2>Portfolio</h2>
          <p className="text-muted">Build your portfolio and run analysis.</p>

          {/* Placeholder blocks */}
          <Portfolio holdings={holdings} />
          <AddHoldingForm holdings={holdings} setHoldings={setHoldings} />
          <div className="panel-block">Run analysis buttons</div>
          <div className="panel-block">Stress test controls</div>
        </section>

        {/* Right: Results */}
        <section className="simulator-panel results-panel">
          <h2>Results</h2>
          <p className="text-muted">
            Equity curve and risk metrics will appear here.
          </p>

          {/* Placeholder blocks */}
          <div className="panel-block">Equity curve</div>
          <div className="panel-block">Metrics</div>
          <div className="panel-block">Scenario deltas</div>
        </section>
      </div>
    </div>
  );
}
