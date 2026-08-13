export function LoadingState({ label = "Loading market evidence" }: { label?: string }) {
  return <div className="state-card"><span className="pulse" />{label}…</div>;
}

export function ErrorState({ message }: { message: string }) {
  return <div className="state-card state-error"><strong>Data unavailable</strong><span>{message}</span></div>;
}
