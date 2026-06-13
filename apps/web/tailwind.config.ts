import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17202a",
        moss: "#4c6b55",
        clay: "#b65c45",
        mist: "#eef3f1",
      },
    },
  },
  plugins: [],
};

export default config;
