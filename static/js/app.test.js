jest.mock('./app.js', () => {
  const originalModule = jest.requireActual('./app.js');
  return {
    __esModule: true,
    ...originalModule,
    // Mock any functions that have side effects if needed
  };
});

beforeEach(() => {
  fetch.resetMocks();
  fetch.mockResponse(JSON.stringify({})); // Default mock for any fetch calls

  // Mock for firebase config
  fetch.mockResponseOnce(JSON.stringify({
    apiKey: "test-api-key",
    authDomain: "test-auth-domain",
    projectId: "test-project-id",
    storageBucket: "test-storage-bucket",
    messagingSenderId: "test-messaging-sender-id",
    appId: "test-app-id"
  }), { url: '/api/firebase-config' });

  document.body.innerHTML = `
    <div id="messages"></div>
    <canvas id="videoCanvas"></canvas>
    <input id="message" />
    <form id="messageForm"></form>
    <button id="sendButton"></button>
    <button id="startAudioButton"></button>
    <button id="startVideoButton"></button>
    <div id="audio-loader"></div>
    <div id="video-pip-container">
        <div class="pip-drag-handle"></div>
    </div>
    <video id="videoPreview"></video>
    <div id="login-gate"></div>
    <button id="signInButton"></button>
    <button id="signOutButton"></button>
    <div id="user-profile"></div>
    <span id="user-name"></span>
    <img id="user-avatar" />
  `;
});

describe('parseAdvancedMarkdown', () => {
  test('should correctly convert bold text', () => {
    const { parseAdvancedMarkdown } = require('./app.js');
    const markdown = 'This is **bold** text.';
    const expectedHtml = '<p class="content-paragraph">This is <strong class="highlight-text">bold</strong> text.</p>';
    const result = parseAdvancedMarkdown(markdown).trim();
    expect(result).toBe(expectedHtml);
  });
});
