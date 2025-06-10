/**
 * app.js: JS code for the adk-streaming sample app.
 */

// WebSocket handling
const sessionId = Math.random().toString().substring(10);

// WebSocket handling
let ws_protocol;
if (window.location.protocol === "https:") {
    ws_protocol = "wss:"; // Use Secure WebSockets if page is HTTPS
} else {
    ws_protocol = "ws:";  // Use regular WebSockets if page is HTTP
}
const ws_url = ws_protocol + "//" + window.location.host + "/ws/" + sessionId;
console.log("Attempting to connect to WebSocket URL:", ws_url);

let websocket = null;
let is_audio_mode_active = false;
let is_video_mode_active = false;

// Get DOM elements
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("message");
const messagesDiv = document.getElementById("messages");
let currentMessageId = null;
let loadingIndicatorId = null; // Track the loading indicator

const audioLoader = document.getElementById('audio-loader');
const startAudioButton = document.getElementById("startAudioButton");

const startVideoButton = document.getElementById("startVideoButton");
const videoPipContainer = document.getElementById("video-pip-container");
const videoPreview = document.getElementById("videoPreview");
const videoCanvas = document.getElementById("videoCanvas");
const videoCtx = videoCanvas.getContext('2d');

let videoStream = null;
let audioStream = null;

let videoFrameInterval = null;
const VIDEO_FRAME_INTERVAL_MS = 1000;
const VIDEO_FRAME_QUALITY = 0.7;

// --- Loading State Control ---
function showAgentThinking(isThinking) {
    if (isThinking) {
        // Remove any existing loading indicator
        if (loadingIndicatorId) {
            const existingLoader = document.getElementById(loadingIndicatorId);
            if (existingLoader) existingLoader.remove();
        }
        
        // Create new loading indicator
        loadingIndicatorId = "loading_" + Math.random().toString(36).substring(7);
        const loadingElem = document.createElement("div");
        loadingElem.id = loadingIndicatorId;
        loadingElem.classList.add("message", "system");
        loadingElem.innerHTML = `
            <div class="loader-dots">
               <div></div><div></div><div></div><div></div>
            </div>
            <span>AVA is processing your request...</span>
        `;
        messagesDiv.appendChild(loadingElem);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } else {
        // Remove loading indicator
        if (loadingIndicatorId) {
            const loadingElem = document.getElementById(loadingIndicatorId);
            if (loadingElem) loadingElem.remove();
            loadingIndicatorId = null;
        }
    }
}

// WebSocket handlers
function connectWebsocket() {
    const connect_in_audio_mode = is_audio_mode_active;
    if (websocket) {
        websocket.close();
    }
    websocket = new WebSocket(ws_url + "?is_audio=" + connect_in_audio_mode);

    websocket.onopen = function () {
        console.log("WebSocket connection opened. Audio response mode:", connect_in_audio_mode);
        appendLog("Connection opened.");
        const welcomeMsg = document.querySelector("#messages .system");
        if (welcomeMsg) welcomeMsg.textContent = "Connection established. Ready for transmission.";
        document.getElementById("sendButton").disabled = false;
        addSubmitHandler();
    };

    websocket.onmessage = function (event) {
        try {
            const message_from_server = JSON.parse(event.data);
            console.log("[AGENT TO CLIENT] ", message_from_server);
            console.log("message_from_server.turn_complete:", message_from_server.turn_complete);
    
            if (message_from_server.turn_complete) {
                currentMessageId = null;
                showAgentThinking(false); // Hide loading indicator
                return;
            }
    
            if (message_from_server.mime_type === "audio/pcm" && audioPlayerNode) {
                // Hide thinking indicator when audio response starts
                if (isProcessingAudioResponse) {
                    showAgentThinking(false);
                    isProcessingAudioResponse = false;
                }
                audioPlayerNode.port.postMessage(base64ToArray(message_from_server.data));
            } else if (message_from_server.mime_type === "text/plain") {
                // Hide loading indicator when first text response arrives
                if (loadingIndicatorId) {
                    showAgentThinking(false);
                }
                
                const textData = message_from_server.data;
                let messageElem = document.getElementById(currentMessageId);
    
                if (!messageElem) {
                    currentMessageId = "msg_" + Math.random().toString(36).substring(7);
                    messageElem = document.createElement("div");
                    messageElem.id = currentMessageId;
                    messageElem.classList.add("message", "remote");
                    messagesDiv.appendChild(messageElem);
                }
    
                const imageUrlRegex = /(\/static\/charts\/[a-zA-Z0-9_-]+\.png)/g;
                let existingText = messageElem.textContent || '';
                let newContent = existingText + textData;
                
                // Convert markdown to HTML
                const htmlContent = parseMarkdownToHTML(newContent);
                
                // Handle image URLs
                const finalContent = htmlContent.replace(
                    imageUrlRegex,
                    '<br><img src="$1" alt="Generated Chart" style="max-width: 100%; display: block; margin: 10px 0; border: 1px solid #ccc; border-radius: 8px;"><br>'
                );
                
                messageElem.innerHTML = finalContent;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        } catch (error) {
            console.error("Error in websocket.onmessage:", error);
        }
    }    

    // Simple markdown to HTML parser function
function parseMarkdownToHTML(markdown) {
    let html = markdown;
    
    // Headers
    html = html.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Inline code
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Unordered lists
    html = html.replace(/^\* (.*$)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    
    // Clean up multiple <br> tags
    html = html.replace(/(<br>\s*){3,}/g, '<br><br>');
    
    return html;
}
    websocket.onclose = function () {
        console.log("WebSocket connection closed.");
        appendLog("Connection lost. Reconnecting in 5s...", "system");
        document.getElementById("sendButton").disabled = true;
        showAgentThinking(false); // Ensure thinking indicator is hidden on close
        // Clear audio timers on connection close
        if (audioSilenceTimer) {
            clearTimeout(audioSilenceTimer);
            audioSilenceTimer = null;
        }
        isProcessingAudioResponse = false;
        setTimeout(connectWebsocket, 5000);
    };

    websocket.onerror = function (e) {
        console.error("WebSocket error: ", e);
        appendLog("WebSocket error.", "system");
    };
}

function appendLog(text, type = "system") {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", type);
    messageDiv.textContent = text;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    if (type === "user") messageInput.value = "";
}

function addSubmitHandler() {
    if (messageForm.onsubmit) return;
    messageForm.onsubmit = function (e) {
        e.preventDefault();
        try {
            const messageText = messageInput.value.trim();
            if (messageText) {
                appendLog(messageText, "user");
                sendMessage({ mime_type: "text/plain", data: messageText });
                showAgentThinking(true); // Show indicator when a message is sent
                console.log("[CLIENT TO AGENT] Text: " + messageText);
            }
        } catch (error) {
            console.error("Error in messageForm.onsubmit:", error);
        }
        return false;
    };
}

function sendMessage(message) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(message));
    } else {
        console.warn("WebSocket not open. Message not sent:", message);
    }
}

function base64ToArray(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

// Audio handling
let audioPlayerNode;
let audioRecorderNode;
let audioSilenceTimer = null;
let isProcessingAudioResponse = false;
import { startAudioPlayerWorklet } from "./audio-player.js";
import { startAudioRecorderWorklet, stopMicrophone as stopAudioCapture } from "./audio-recorder.js";

async function toggleAudio() {
    console.log("toggleAudio function called. is_audio_mode_active:", is_audio_mode_active);
    startAudioButton.disabled = true;
    if (!is_audio_mode_active) {
        audioLoader.classList.remove('hidden');
        try {
            if (!audioPlayerNode) {
                [audioPlayerNode] = await startAudioPlayerWorklet();
            }
            [audioRecorderNode, , audioStream] = await startAudioRecorderWorklet(audioRecorderHandler);
            is_audio_mode_active = true;
            startAudioButton.classList.add('active');
            appendLog("Voice chat activated.", "system");
            if (websocket && websocket.readyState === WebSocket.OPEN && !websocket.url.includes("is_audio=true")) {
                websocket.close();
            } else if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                connectWebsocket();
            }
        } catch (error) {
            console.error("Error starting audio:", error);
            appendLog("Error starting audio.", "system");
            startAudioButton.classList.remove('active');
        } finally {
            audioLoader.classList.add('hidden');
            startAudioButton.disabled = false;
        }
    } else {
        if (audioRecorderNode && audioStream) {
            stopAudioCapture(audioStream);
        }
        // Clear any pending timers
        if (audioSilenceTimer) {
            clearTimeout(audioSilenceTimer);
            audioSilenceTimer = null;
        }
        is_audio_mode_active = false;
        isProcessingAudioResponse = false;
        startAudioButton.classList.remove('active');
        appendLog("Voice chat deactivated.", "system");
        startAudioButton.disabled = false;
        if (websocket && websocket.readyState === WebSocket.OPEN && websocket.url.includes("is_audio=true")) {
            websocket.close();
        }
    }
}

if (startAudioButton) startAudioButton.addEventListener("click", toggleAudio);

function audioRecorderHandler(pcmData) {
    if (!is_audio_mode_active) return;
    
    // Clear any existing silence timer
    if (audioSilenceTimer) {
        clearTimeout(audioSilenceTimer);
        audioSilenceTimer = null;
    }
    
    // Check if this is actual audio data (not silence)
    const audioBuffer = new Uint8Array(pcmData);
    let hasAudio = false;
    for (let i = 0; i < audioBuffer.length; i += 2) {
        const sample = Math.abs((audioBuffer[i] | (audioBuffer[i + 1] << 8)) - 32768);
        if (sample > 1000) { // Threshold for detecting speech
            hasAudio = true;
            break;
        }
    }
    
    if (hasAudio) {
        // Set timer to detect when user stops speaking
        audioSilenceTimer = setTimeout(() => {
            if (is_audio_mode_active && !isProcessingAudioResponse) {
                showAgentThinking(true);
                isProcessingAudioResponse = true;
                console.log("Audio silence detected - showing thinking indicator");
            }
        }, 1500); // Show thinking indicator after 1.5 seconds of silence
    }
    
    sendMessage({ mime_type: "audio/pcm", data: arrayBufferToBase64(pcmData) });
}

function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

// Video Handling
async function toggleVideo() {
    if (!is_video_mode_active) {
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoPreview.srcObject = videoStream;
            videoPipContainer.classList.remove('hidden');
            if (videoFrameInterval) clearInterval(videoFrameInterval);
            videoFrameInterval = setInterval(sendVideoFrame, VIDEO_FRAME_INTERVAL_MS);
            is_video_mode_active = true;
            startVideoButton.classList.add('active');
            appendLog("Video feed started.", "system");
        } catch (err) {
            console.error("Error accessing webcam:", err);
            appendLog("Error starting video: " + err.message, "system");
        }
    } else {
        if (videoStream) videoStream.getTracks().forEach(track => track.stop());
        videoPipContainer.classList.add('hidden');
        if (videoFrameInterval) clearInterval(videoFrameInterval);
        is_video_mode_active = false;
        startVideoButton.classList.remove('active');
        appendLog("Video feed stopped.", "system");
    }
}

function sendVideoFrame() {
    if (!is_video_mode_active || !videoStream || videoPreview.paused || videoPreview.ended || !videoPreview.videoWidth) return;
    videoCanvas.width = videoPreview.videoWidth;
    videoCanvas.height = videoPreview.videoHeight;
    videoCtx.drawImage(videoPreview, 0, 0, videoCanvas.width, videoCanvas.height);
    const imageDataUrl = videoCanvas.toDataURL('image/jpeg', VIDEO_FRAME_QUALITY);
    const base64ImageData = imageDataUrl.split(',')[1];
    if (base64ImageData) {
        sendMessage({ mime_type: "image/jpeg", data: base64ImageData });
    }
}

if (startVideoButton) startVideoButton.addEventListener("click", toggleVideo);

document.addEventListener('DOMContentLoaded', connectWebsocket);

// --- Draggable PiP Widget Logic ---
const pipContainer = document.getElementById('video-pip-container');
const dragHandle = document.querySelector('.pip-drag-handle');

let isDragging = false;
let offsetX, offsetY;

// Unified Drag Start for both Mouse and Touch
const dragStart = (e) => {
  isDragging = true;
  document.body.classList.add('pip-dragging');
  pipContainer.classList.add('pip-dragging');

  // Calculate offset from the element's top-left corner
  const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
  const clientY = e.type === 'touchstart' ? e.touches[0].clientY : e.clientY;
  
  const rect = pipContainer.getBoundingClientRect();
  offsetX = clientX - rect.left;
  offsetY = clientY - rect.top;
};

// Unified Drag Move
const dragMove = (e) => {
  if (!isDragging) return;
  e.preventDefault(); // Prevent scrolling on mobile

  const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
  const clientY = e.type === 'touchmove' ? e.touches[0].clientY : e.clientY;

  // Calculate new position
  let newX = clientX - offsetX;
  let newY = clientY - offsetY;

  // Constrain to viewport boundaries
  const containerWidth = pipContainer.offsetWidth;
  const containerHeight = pipContainer.offsetHeight;
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;

  newX = Math.max(0, Math.min(newX, viewportWidth - containerWidth));
  newY = Math.max(0, Math.min(newY, viewportHeight - containerHeight));

  // Apply the new position
  pipContainer.style.left = `${newX}px`;
  pipContainer.style.top = `${newY}px`;

  // Important: Remove bottom/right positioning if it was set initially
  pipContainer.style.right = 'auto';
  pipContainer.style.bottom = 'auto';
};

// Unified Drag End
const dragEnd = () => {
  isDragging = false;
  document.body.classList.remove('pip-dragging');
  pipContainer.classList.remove('pip-dragging');
};

// Attach Event Listeners
dragHandle.addEventListener('mousedown', dragStart);
dragHandle.addEventListener('touchstart', dragStart);

document.addEventListener('mousemove', dragMove);
document.addEventListener('touchmove', dragMove, { passive: false }); // passive:false is needed for e.preventDefault() to work

document.addEventListener('mouseup', dragEnd);
document.addEventListener('touchend', dragEnd);