import Link from "next/link";

export default function NotFound() {
  return (
    <div className="max-w-xl py-16">
      <h1 className="font-mono text-2xl font-bold text-ink">Not found</h1>
      <p className="mt-2 text-sm text-body">
        That ticker is not in the database. It may not be a US SEC filer, or it
        may fall below the revenue floor used when loading the universe.
      </p>
      <p className="mt-4">
        <Link href="/" className="text-accent underline decoration-dotted">
          Back to the screener
        </Link>
      </p>
    </div>
  );
}
