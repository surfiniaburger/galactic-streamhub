/**
 * app.js: JS code for the adk-streaming sample app.
 */

// WebSocket handling
const sessionId = Math.random().toString().substring(10);

// ---- THIS IS THE CRUCIAL MODIFICATION ----
let ws_protocol;
if (window.location.protocol === "https:") {
    ws_protocol = "wss:"; // Use Secure WebSockets if page is HTTPS
} else {
    ws_protocol = "ws:";  // Use regular WebSockets if page is HTTP
}
const ws_url = ws_protocol + "//" + window.location.host + "/ws/" + sessionId;
// ---- END MODIFICATION ----

// For debugging, let's log the URL to be absolutely sure:
console.log("Attempting to connect to WebSocket URL:", ws_url); // ADD THIS LINE FOR DEBUGGING


let websocket = null;
let is_audio_mode_active = false;
let is_video_mode_active = false;

// Get DOM elements
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("message");
const messagesDiv = document.getElementById("messages");
let currentMessageId = null;

const audioLoader = document.getElementById('audio-loader'); // Get the loader element
const startAudioButton = document.getElementById("startAudioButton");
const audioButtonTextFace = startAudioButton.querySelector('.button-top-face'); // For 3D button text

const startVideoButton = document.getElementById("startVideoButton");
const videoButtonTextFace = startVideoButton.querySelector('.button-top-face'); // For 3D button text

const videoPreview = document.getElementById("videoPreview");
const videoCanvas = document.getElementById("videoCanvas");
const videoCtx = videoCanvas.getContext('2d');

const mediaDisplayPanel = document.getElementById('media-display-panel'); // Get the panel

let localMediaStream = null;
let videoStream = null;
let audioStream = null;

let videoFrameInterval = null;
const VIDEO_FRAME_INTERVAL_MS = 1000;
const VIDEO_FRAME_QUALITY = 0.7;

// WebSocket handlers
function connectWebsocket() {
    const connect_in_audio_mode = is_audio_mode_active;
    websocket = new WebSocket(ws_url + "?is_audio=" + connect_in_audio_mode);

    websocket.onopen = function () {
        console.log("WebSocket connection opened. Audio response mode:", connect_in_audio_mode);
        appendLog("Connection opened.");
        document.getElementById("sendButton").disabled = false;
        addSubmitHandler();
    };

    websocket.onmessage = function (event) {
        const message_from_server = JSON.parse(event.data);
        console.log("[AGENT TO CLIENT] ", message_from_server);

        if (message_from_server.turn_complete) {
            currentMessageId = null;
            return;
        }

        if (message_from_server.mime_type === "audio/pcm" && audioPlayerNode) {
            audioPlayerNode.port.postMessage(base64ToArray(message_from_server.data));
        } else if (message_from_server.mime_type === "text/plain") {
            let messageElem = document.getElementById(currentMessageId);
            if (!messageElem) {
                currentMessageId = "msg_" + Math.random().toString(36).substring(7);
                messageElem = document.createElement("div"); // Should be 'div' to match your HTML structure for messages
                messageElem.id = currentMessageId;
                messageElem.classList.add("message", "remote"); // Use consistent class with HTML
                messagesDiv.appendChild(messageElem);
            }
            messageElem.textContent += message_from_server.data;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    };

    websocket.onclose = function () {
        console.log("WebSocket connection closed.");
        appendLog("Connection closed. Reconnecting in 5s...");
        document.getElementById("sendButton").disabled = true;
        setTimeout(connectWebsocket, 5000);
    };

    websocket.onerror = function (e) {
        console.error("WebSocket error: ", e);
        appendLog("WebSocket error.");
    };
}




function appendLog(text, type = "system") {
    const messageDiv = document.createElement("div"); // Changed from 'p' to 'div'
    messageDiv.classList.add("message", type); // e.g. "message system", "message user"
    messageDiv.textContent = text; // No need for "You: " prefix if styled by class
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    if (type === "user") messageInput.value = "";
}


function addSubmitHandler() {
    if (messageForm.onsubmit) return;
    messageForm.onsubmit = function (e) {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (messageText) {
            appendLog(messageText, "user"); // Using new appendLog structure
            sendMessage({
                mime_type: "text/plain",
                data: messageText,
            });
            console.log("[CLIENT TO AGENT] Text: " + messageText);
        }
        return false;
    };
}

function sendMessage(message) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        const messageJson = JSON.stringify(message);
        websocket.send(messageJson);
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

/**
 * Audio handling
 */
let audioPlayerNode;
let audioRecorderNode;

import { startAudioPlayerWorklet } from "./audio-player.js";
import { startAudioRecorderWorklet, stopMicrophone as stopAudioCapture } from "./audio-recorder.js";

async function toggleAudio() {
    console.log("toggleAudio function called. is_audio_mode_active:", is_audio_mode_active);

    if (!is_audio_mode_active) { // ---- If turning ON audio ----
        // Disable button and show loader
        if (startAudioButton) startAudioButton.disabled = true;
        if (audioButtonTextFace) audioButtonTextFace.textContent = "Initializing..."; // Optional: change button text
        if (audioLoader) audioLoader.classList.remove('hidden');

        try {
            if (!audioPlayerNode) {
                const [playerNode, playerCtx] = await startAudioPlayerWorklet();
                audioPlayerNode = playerNode;
            }
            const [recorderNode, recorderCtx, stream] = await startAudioRecorderWorklet(audioRecorderHandler);
            audioRecorderNode = recorderNode;
            audioStream = stream;
            is_audio_mode_active = true;

            if (audioButtonTextFace) audioButtonTextFace.textContent = "Stop Audio";
            appendLog("Audio input started.", "system");

            if (websocket && websocket.readyState === WebSocket.OPEN && websocket.url.includes("is_audio=false")) {
                 console.log("WebSocket open but in text response mode. Reconnecting for audio responses.");
                 websocket.close();
            } else if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                connectWebsocket();
            }
        } catch (error) {
            console.error("Error starting audio:", error);
            appendLog("Error starting audio.", "system");
            is_audio_mode_active = false; // Ensure state is correct on error
            if (audioButtonTextFace) audioButtonTextFace.textContent = "Start Audio"; // Reset button text
        } finally {
            // Always hide loader and re-enable button
            if (audioLoader) audioLoader.classList.add('hidden');
            if (startAudioButton) startAudioButton.disabled = false;
            // If error occurred, button text is already reset. If success, it's "Stop Audio".
            // If it was "Initializing..." and an error happened before "Stop Audio" was set,
            // we need to ensure it goes back to "Start Audio".
            if (!is_audio_mode_active && audioButtonTextFace && audioButtonTextFace.textContent !== "Start Audio") {
                audioButtonTextFace.textContent = "Start Audio";
            }
        }
    } else { // ---- If turning OFF audio ----
        // This part is quick, so typically no loader needed here
        if (audioRecorderNode && audioStream) {
            stopAudioCapture(audioStream);
            audioRecorderNode.disconnect();
            audioRecorderNode = null;
            audioStream = null;
        }
        is_audio_mode_active = false;
        if (audioButtonTextFace) audioButtonTextFace.textContent = "Start Audio";
        appendLog("Audio input stopped.", "system");
        if (websocket && websocket.readyState === WebSocket.OPEN && websocket.url.includes("is_audio=true")) {
            console.log("WebSocket open but in audio response mode. Reconnecting for text responses.");
            websocket.close();
        }
    }
}


if (startAudioButton) {
    startAudioButton.addEventListener("click", toggleAudio);
} else {
    console.error("Start Audio Button not found");
}


function audioRecorderHandler(pcmData) {
    if (!is_audio_mode_active) return;
    sendMessage({
        mime_type: "audio/pcm",
        data: arrayBufferToBase64(pcmData),
    });
}

function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

/**
 * NEW: Video Handling with Panel Toggle
 */
async function toggleVideo() {
  console.log("toggleVideo function called. is_video_mode_active:", is_video_mode_active);
    if (!is_video_mode_active) { // Turning ON video
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoStream = stream;
            videoPreview.srcObject = stream;
            videoPreview.play();

            // ---- SHOW PANEL ----
            if (mediaDisplayPanel) mediaDisplayPanel.classList.remove('hidden-animated');

            if (videoFrameInterval) clearInterval(videoFrameInterval);
            videoFrameInterval = setInterval(sendVideoFrame, VIDEO_FRAME_INTERVAL_MS);

            is_video_mode_active = true;
            if (videoButtonTextFace) videoButtonTextFace.textContent = "Stop Video"; // Updated
            appendLog("Video capture started.", "system");
        } catch (err) {
            console.error("Error accessing webcam:", err);
            appendLog("Error starting video: " + err.message, "system");
            is_video_mode_active = false;
            if (videoButtonTextFace) videoButtonTextFace.textContent = "Start Video"; // Updated
            // ---- HIDE PANEL ON ERROR ----
            if (mediaDisplayPanel && !mediaDisplayPanel.classList.contains('hidden-animated')) {
                mediaDisplayPanel.classList.add('hidden-animated');
            }
        }
    } else { // Turning OFF video
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
        videoPreview.srcObject = null;
        if (videoFrameInterval) {
            clearInterval(videoFrameInterval);
            videoFrameInterval = null;
        }
        // ---- HIDE PANEL ----
        if (mediaDisplayPanel) mediaDisplayPanel.classList.add('hidden-animated');

        is_video_mode_active = false;
        if (videoButtonTextFace) videoButtonTextFace.textContent = "Start Video"; // Updated
        appendLog("Video capture stopped.", "system");
    }
}

function sendVideoFrame() {
    if (!is_video_mode_active || !videoStream || videoPreview.paused || videoPreview.ended || !videoPreview.videoWidth) {
        return;
    }
    videoCanvas.width = videoPreview.videoWidth;
    videoCanvas.height = videoPreview.videoHeight;
    videoCtx.drawImage(videoPreview, 0, 0, videoCanvas.width, videoCanvas.height);
    const imageDataUrl = videoCanvas.toDataURL('image/jpeg', VIDEO_FRAME_QUALITY);
    const base64ImageData = imageDataUrl.split(',')[1];

    if (base64ImageData) {
        sendMessage({
            mime_type: "image/jpeg",
            data: base64ImageData
        });
    }
}

if (startVideoButton) {
    startVideoButton.addEventListener("click", toggleVideo);
} else {
    console.error("Start Video Button not found");
}


// Initial connection
// Ensure DOM is fully loaded before trying to connect or add listeners,
// especially if script is in <head> or not type="module" which defers.
// However, since this is a module, it should be fine.
document.addEventListener('DOMContentLoaded', () => {
    // Check if elements exist before calling connectWebsocket or other functions
    // that depend on them if there were any issues.
    // For now, assuming all critical elements are present for connectWebsocket.
    connectWebsocket();

    // Ensure the media display panel is hidden initially if it wasn't by CSS default (it should be)
    if (mediaDisplayPanel && !mediaDisplayPanel.classList.contains('hidden-animated')) {
         // This is belt-and-suspenders; CSS should handle initial state.
         // mediaDisplayPanel.classList.add('hidden');
    }

    // If you have other initializations that depend on the DOM, put them here.
});