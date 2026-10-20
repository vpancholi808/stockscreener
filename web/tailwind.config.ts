import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F5F7FA",
        panel: "#FFFFFF",
        ink: "#0E1726",
        body: "#3A4759",
        muted: "#6B7A8F",
        rule: "#DCE3EC",
        pass: "#0B6E5F",
        fail: "#9E3B3B",
        warn: "#8A6318",
        accent: "#243E6B",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
    },
  },
  plugins: [],
};
export default config;
