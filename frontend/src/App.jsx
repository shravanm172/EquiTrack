import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import SimulatorPage from "./pages/Simulator";

function App() {
  const [count, setCount] = useState(0);

  return <SimulatorPage />;
}

export default App;
