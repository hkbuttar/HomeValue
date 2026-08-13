import type { Metadata } from "next";
import { Header } from "@/components/header";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "HomeValue — Chicago housing intelligence", template: "%s · HomeValue" },
  description: "Explainable Chicago home valuations grounded in property, place, and market evidence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en"><body>
      <Header /><main>{children}</main>
      <footer><span>HomeValue</span><p>Chicago housing valuation with uncertainty made visible.</p></footer>
    </body></html>
  );
}
