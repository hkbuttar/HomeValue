import Link from "next/link";
import { MarketMap } from "@/components/market-map";

export default function Home() {
  return (
    <>
      <section className="hero shell">
        <div className="hero-copy">
          <p className="eyebrow">Chicago housing, read in context</p>
          <h1>A home is a structure.<br/><em>Its value is a place.</em></h1>
          <p className="lede">HomeValue combines property facts, neighborhood movement, accessibility, spatial markets, and prior comparable sales—then shows its uncertainty.</p>
          <div className="button-row"><Link className="button" href="/valuation">Estimate a value <span>→</span></Link><Link className="text-link" href="/research">Read the evidence</Link></div>
          <div className="trust-row"><span>Out-of-time tested</span><span>Spatially validated</span><span>Calibrated ranges</span></div>
        </div>
        <div className="hero-map"><MarketMap/><div className="map-caption"><strong>77 markets</strong><span>one connected housing system</span></div></div>
      </section>
      <section className="manifesto shell">
        <p className="section-number">01 / THE MODEL</p><h2>Not just a number.<br/>A defensible explanation.</h2>
        <div className="principle-grid">
          <article><span>PROPERTY</span><h3>What you own</h3><p>Size, age, rooms, construction, land, and the physical attributes buyers compare.</p></article>
          <article><span>PLACE</span><h3>Where it belongs</h3><p>Neighborhood trajectory, transit, lake access, urban form, and nearby market signals.</p></article>
          <article><span>MARKET</span><h3>When it trades</h3><p>Time-aware pricing trained on yesterday and tested against genuinely later sales.</p></article>
        </div>
      </section>
      <section className="evidence-band"><div className="shell evidence-inner"><p>Every estimate arrives with</p><div><strong>A likely range</strong><strong>Comparable evidence</strong><strong>Value drivers</strong><strong>Known limitations</strong></div></div></section>
    </>
  );
}
