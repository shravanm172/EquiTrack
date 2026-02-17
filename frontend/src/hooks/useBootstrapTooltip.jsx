import { useEffect } from "react";
import { Tooltip } from "bootstrap";

export default function useBootstrapTooltips(deps = []) {
  useEffect(() => {
    const tooltipTriggerList = document.querySelectorAll(
      '[data-bs-toggle="tooltip"]',
    );

    const tooltipList = Array.from(tooltipTriggerList).map(
      (tooltipTriggerEl) => new Tooltip(tooltipTriggerEl),
    );

    return () => {
      tooltipList.forEach((tooltip) => tooltip.dispose());
    };
  }, deps);
}
