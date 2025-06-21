
// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html", // Path to your main HTML file
    // Add other paths if you have separate JS files that dynamically add classes
    // "./src/**/*.{js,ts,jsx,tsx}", // Example for React/Vue/other JS frameworks
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}