
module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['**/static/js/**/*.test.js'],
  setupFilesAfterEnv: ['./jest.setup.js'],
  moduleNameMapper: {
    '^https://www.gstatic.com/firebasejs/12.4.0/firebase-app.js$': 'firebase/app',
    '^https://www.gstatic.com/firebasejs/12.4.0/firebase-analytics.js$': 'firebase/analytics',
    '^https://www.gstatic.com/firebasejs/12.4.0/firebase-auth.js$': 'firebase/auth',
    '^https://www.gstatic.com/firebasejs/12.4.0/firebase-app-check.js$': 'firebase/app-check',
    './audio-player.js': '<rootDir>/__mocks__/fileMock.js',
    './audio-recorder.js': '<rootDir>/__mocks__/fileMock.js'
  },
};
