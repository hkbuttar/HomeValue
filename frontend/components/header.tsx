import Link from "next/link";

const links = [
  ["Market", "/market"],
  ["Neighborhoods", "/neighborhoods"],
  ["Spatial lab", "/spatial-lab"],
  ["Research", "/research"],
];

export function Header() {
  return (
    <header className="site-header">
      <Link className="brand" href="/" aria-label="HomeValue home">
        <span className="brand-mark">HV</span>
        <span>HomeValue</span>
      </Link>
      <nav aria-label="Primary navigation">
        {links.map(([label, href]) => <Link key={href} href={href}>{label}</Link>)}
      </nav>
      <Link className="button button-small" href="/valuation">Value a home</Link>
    </header>
  );
}
