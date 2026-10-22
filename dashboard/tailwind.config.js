/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        space: {
          900: '#0a0d14',
          800: '#101522',
          700: '#182032',
          600: '#222d46',
          500: '#334166',
        },
        ice: {
          400: '#67e8f9',
          500: '#06b6d4',
          600: '#0891b2',
        }
      }
    },
  },
  plugins: [],
}
