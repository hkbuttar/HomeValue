import { ResearchPayload } from "@/components/record-view";

export default function SpatialLabPage() {
  return <section className="shell"><div className="page-intro"><p className="eyebrow">Spatial lab</p><h1>What ordinary regression leaves on the map.</h1><p>Residual clustering, neighboring-price dependence, omitted local conditions, and the limits of geographic transfer.</p></div>
    <div className="research-grid"><ResearchPayload path="/models/spatial" title="Spatial model evidence"/><ResearchPayload path="/accessibility/transit" title="CTA robustness"/><ResearchPayload path="/accessibility/lake" title="Lake and downtown gradients"/></div>
  </section>;
}
