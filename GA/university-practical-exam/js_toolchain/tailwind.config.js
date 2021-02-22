module.exports = {
  purge: {
    enabled: true,
    content: ['../practical_exam/exam_scheduling/templates/**/*.html'],
    options: {
      safelist: ['errorlist'],
    }
  },
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
    },
  },
  variants: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
