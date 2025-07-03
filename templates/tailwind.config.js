// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./templates/index.html",
    "./templates/**/*.html",
    "./index.html",
    "./templates/test.html",
    // "./*.html", // to include all HTML files in the root
    "./js/**/*.js", // to include all JS files in a 'js' folder
    "./src/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // Add a new custom color or override an existing one
        // Example: 'emerald' is a Tailwind default.
        // You could change the green-500 to a custom hex value here.
        'primary-green': '#00B050', // A custom green
        'custom-blue': '#1a73e8', // A custom blue
      },
      // You can also extend spacing, fontSize, etc.
    },
  },
  plugins: [],
}
