export default function InfoTooltip({ text }) {
  return (
    <span
      className="ms-1 my-text-muted"
      data-bs-toggle="tooltip"
      data-bs-placement="top"
      title={text}
      style={{ cursor: "pointer", fontSize: "0.9rem" }}
    >
      ⓘ
    </span>
  );
}
