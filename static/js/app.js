/**
 * app.js: JS code for the adk-streaming sample app, integrated with Firebase.
 */

// Import the new Firebase functionalities
// This assumes you have created the 'firebase-chat.js' file in the same directory.
import { sendChatMessage, setAgentThinking } from './firebase-chat.js';

// --- WebSocket Handling ---
const sessionId = Math.random().toString(36).substring(2);
let ws_protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const ws_url = ws_protocol + "//" + window.location.host + "/ws/" + sessionId;
console.log("Attempting to connect to WebSocket URL:", ws_url);

let websocket = null;
let is_audio_mode_active = false;
let is_video_mode_active = false;

// --- DOM Element Selection ---
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("message");

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


// --- WebSocket Connection Logic ---
function connectWebsocket() {
    websocket = new WebSocket(ws_url + "?is_audio=" + is_audio_mode_active);

    websocket.onopen = () => {
        console.log("WebSocket connection opened.");
        const welcomeMsg = document.querySelector("#messages .system");
        if (welcomeMsg) welcomeMsg.textContent = "Connection established. Ready for transmission.";
        document.getElementById("sendButton").disabled = false;
    };

    websocket.onclose = () => {
        console.log("WebSocket connection closed.");
        // We no longer append logs directly, but you could set a system status via Firebase if needed
        document.getElementById("sendButton").disabled = true;
        setTimeout(connectWebsocket, 5000);
    };

    websocket.onerror = (e) => console.error("WebSocket error: ", e);

    websocket.onmessage = function (event) {
        setAgentThinking(false); // Agent has responded, hide "thinking" indicator
        
        const message_from_server = JSON.parse(event.data);
        console.log("[AGENT TO CLIENT] ", message_from_server);

        if (message_from_server.turn_complete) return;

        if (message_from_server.mime_type === "audio/pcm" && audioPlayerNode) {
            audioPlayerNode.port.postMessage(base64ToArray(message_from_server.data));
        } else if (message_from_server.mime_type === "text/plain") {
            // Send the agent's final text response to be displayed via Firestore
            sendChatMessage(message_from_server.data);
        }
    };
}


// --- Form Submission ---
function addSubmitHandler() {
    if (messageForm.onsubmit) return;
    messageForm.onsubmit = async function (e) {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (messageText) {
            // 1. Send user message to Firestore to display it immediately
            await sendChatMessage(messageText);
            // 2. Set agent status to "thinking" via Realtime DB
            setAgentThinking(true);
            // 3. Send the message to the agent via WebSocket
            sendMessageToAgent({ mime_type: "text/plain", data: messageText });
            
            messageInput.value = ""; // Clear input
            console.log("[CLIENT TO AGENT] Text: " + messageText);
        }
        return false;
    };
}

function sendMessageToAgent(message) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(message));
    } else {
        console.warn("WebSocket not open. Message not sent to agent:", message);
    }
}


// --- Utility Functions ---
function base64ToArray(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}


// --- Audio Handling ---
let audioPlayerNode;
let audioRecorderNode;
import { startAudioPlayerWorklet } from "./audio-player.js";
import { startAudioRecorderWorklet, stopMicrophone as stopAudioCapture } from "./audio-recorder.js";

async function toggleAudio() {
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
        } catch (error) {
            console.error("Error starting audio:", error);
            startAudioButton.classList.remove('active');
        } finally {
            audioLoader.classList.add('hidden');
            startAudioButton.disabled = false;
        }
    } else {
        if (audioRecorderNode && audioStream) {
            stopAudioCapture(audioStream);
        }
        is_audio_mode_active = false;
        startAudioButton.classList.remove('active');
        startAudioButton.disabled = false;
    }
}
if (startAudioButton) startAudioButton.addEventListener("click", toggleAudio);

function audioRecorderHandler(pcmData) {
    if (!is_audio_mode_active) return;
    sendMessageToAgent({ mime_type: "audio/pcm", data: arrayBufferToBase64(pcmData) });
}


// --- Video Handling ---
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
        } catch (err) {
            console.error("Error accessing webcam:", err);
        }
    } else {
        if (videoStream) videoStream.getTracks().forEach(track => track.stop());
        videoPipContainer.classList.add('hidden');
        if (videoFrameInterval) clearInterval(videoFrameInterval);
        is_video_mode_active = false;
        startVideoButton.classList.remove('active');
    }
}
if (startVideoButton) startVideoButton.addEventListener("click", toggleVideo);

function sendVideoFrame() {
    if (!is_video_mode_active || !videoStream || videoPreview.paused || videoPreview.ended || !videoPreview.videoWidth) return;
    videoCanvas.width = videoPreview.videoWidth;
    videoCanvas.height = videoPreview.videoHeight;
    videoCtx.drawImage(videoPreview, 0, 0, videoCanvas.width, videoCanvas.height);
    const imageDataUrl = videoCanvas.toDataURL('image/jpeg', VIDEO_FRAME_QUALITY);
    const base64ImageData = imageDataUrl.split(',')[1];
    if (base64ImageData) {
        sendMessageToAgent({ mime_type: "image/jpeg", data: base64ImageData });
    }
}


// --- Draggable PiP Widget Logic ---
const dragHandle = document.querySelector('.pip-drag-handle');
let isDragging = false;
let offsetX, offsetY;

const dragStart = (e) => {
  isDragging = true;
  document.body.classList.add('pip-dragging');
  videoPipContainer.classList.add('pip-dragging');
  const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
  const clientY = e.type === 'touchstart' ? e.touches[0].clientY : e.clientY;
  const rect = videoPipContainer.getBoundingClientRect();
  offsetX = clientX - rect.left;
  offsetY = clientY - rect.top;
};

const dragMove = (e) => {
  if (!isDragging) return;
  e.preventDefault();
  const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
  const clientY = e.type === 'touchmove' ? e.touches[0].clientY : e.clientY;
  let newX = clientX - offsetX;
  let newY = clientY - offsetY;
  const containerWidth = videoPipContainer.offsetWidth;
  const containerHeight = videoPipContainer.offsetHeight;
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  newX = Math.max(0, Math.min(newX, viewportWidth - containerWidth));
  newY = Math.max(0, Math.min(newY, viewportHeight - containerHeight));
  videoPipContainer.style.left = `${newX}px`;
  videoPipContainer.style.top = `${newY}px`;
  videoPipContainer.style.right = 'auto';
  videoPipContainer.style.bottom = 'auto';
};

const dragEnd = () => {
  isDragging = false;
  document.body.classList.remove('pip-dragging');
  videoPipContainer.classList.remove('pip-dragging');
};

dragHandle.addEventListener('mousedown', dragStart);
dragHandle.addEventListener('touchstart', dragStart, { passive: true });
document.addEventListener('mousemove', dragMove);
document.addEventListener('touchmove', dragMove, { passive: false });
document.addEventListener('mouseup', dragEnd);
document.addEventListener('touchend', dragEnd);


// --- Initial Load ---
document.addEventListener('DOMContentLoaded', () => {
    addSubmitHandler();
    connectWebsocket();
});