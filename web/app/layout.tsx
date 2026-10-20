import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

/** Typography note: the MONO face carries the wordmark, headers and every
 *  figure, while the sans is reserved for prose. Inverting the usual hierarchy
 *  is deliberate — the interface should read as an instrument, not a brochure.
 *
 *  Both stacks are system fonts (see globals.css), so there is no third-party
 *  font request at runtime: nothing to leak to a CDN, no flash of unstyled
 *  text, and no build-time network dependency. To swap in a webfont, add
 *  `next/font/google` here and point --font-sans / --font-mono at it. */

export const metadata: Metadata = {
  title: "finmodel — equity health screener",
  description:
    "Financial health screening computed from SEC filings: 12-metric scorecard, forensic accounting scores, and sector-relative ranking.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans">
        <header className="border-b-2 border-ink bg-panel">
          <div className="mx-auto flex max-w-[1240px] flex-wrap items-center justify-between gap-3 px-4 py-3.5 sm:px-6">
            <Link href="/" className="group flex items-baseline gap-2.5">
              <span className="font-mono text-lg font-bold tracking-tight text-ink">
                finmodel
              </span>
              <span className="font-mono text-2xs uppercase tracking-[0.18em] text-muted">
                equity health screener
              </span>
            </Link>
            <nav className="flex items-center gap-4 text-sm">
              <Link href="/" className="text-body hover:text-ink">
                Screener
              </Link>
              <Link href="/portfolio" className="text-body hover:text-ink">
                Portfolio
              </Link>
              <Link href="/methodology" className="text-body hover:text-ink">
                Methodology
              </Link>
            </nav>
          </div>
        </header>

        <main className="mx-auto max-w-[1240px] px-4 pb-20 pt-6 sm:px-6">
          {children}
        </main>

        <footer className="border-t border-rule">
          <div className="mx-auto max-w-[1240px] px-4 py-6 text-2xs leading-relaxed text-muted sm:px-6">
            <p>
              Computed from public filings with the U.S. Securities and Exchange
              Commission. Figures are as last filed and may be out of date.
            </p>
            <p className="mt-1.5 font-medium text-body">
              This is analysis tooling, not investment advice. Nothing here is a
              recommendation to buy or sell any security. Verify every figure
              against primary sources before making any financial decision.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
