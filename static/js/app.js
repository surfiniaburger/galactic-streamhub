/**
 * app.js: JS code for the Galactic StreamHub app with Firebase Auth.
 */

// --- Firebase Initialization ---
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.9.0/firebase-app.js";
import { getAnalytics, logEvent } from "https://www.gstatic.com/firebasejs/11.9.0/firebase-analytics.js";
import { 
    getAuth, 
    onAuthStateChanged, 
    GoogleAuthProvider, 
    signInWithPopup,
    signOut
} from "https://www.gstatic.com/firebasejs/11.9.0/firebase-auth.js";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "",
  authDomain: "studio-l13dd.firebaseapp.com",
  projectId: "studio-l13dd",
  storageBucket: "studio-l13dd.firebasestorage.app",
  messagingSenderId: "1074728173827",
  appId: "1:1074728173827:web:004b76b81cf68b38bbe936"
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const analytics = getAnalytics(firebaseApp);
const auth = getAuth(firebaseApp);
const provider = new GoogleAuthProvider();

// --- Global State ---
let websocket = null;
let is_audio_mode_active = false;
let is_video_mode_active = false;
let currentMessageId = null;
let loadingIndicatorId = null;
let videoStream = null;
let audioStream = null;
let videoFrameInterval = null;
const VIDEO_FRAME_INTERVAL_MS = 1000;
const VIDEO_FRAME_QUALITY = 0.7;


// --- DOM Elements ---
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("message");
const messagesDiv = document.getElementById("messages");
const sendButton = document.getElementById("sendButton");
const startAudioButton = document.getElementById("startAudioButton");
const startVideoButton = document.getElementById("startVideoButton");
const audioLoader = document.getElementById('audio-loader');
const videoPipContainer = document.getElementById("video-pip-container");
const videoPreview = document.getElementById("videoPreview");
const videoCanvas = document.getElementById("videoCanvas");
const videoCtx = videoCanvas.getContext('2d');

// Auth UI Elements
const loginGate = document.getElementById('login-gate');
const signInButton = document.getElementById('signInButton');
const signOutButton = document.getElementById('signOutButton');
const userProfileDiv = document.getElementById('user-profile');
const userNameSpan = document.getElementById('user-name');
const userAvatarImg = document.getElementById('user-avatar');


// --- Core Authentication Logic ---

// The onAuthStateChanged observer is the central point of control.
onAuthStateChanged(auth, async (user) => {
    if (user) {
        // User is signed in
        console.log("User authenticated:", user.displayName);
        userNameSpan.textContent = user.displayName;
        userAvatarImg.src = user.photoURL;
        userProfileDiv.classList.remove('hidden');
        loginGate.style.display = 'none'; // Hide login gate

        // Get the secure ID token and then connect
        try {
            const idToken = await user.getIdToken(true);
            console.log("ID Token obtained successfully");
            // This ensures we ONLY connect after the token is retrieved.
            setInputsLocked(false); // Unlock the UI
            connectWebsocket(idToken); // Connect with the secure token
        } catch (error) {
            console.error("Error getting ID token:", error);
            setInputsLocked(true);
            appendLog("Authentication token error. Please sign in again.", "system");
        }

    } else {
        // User is signed out
        console.log("User is signed out.");
        userProfileDiv.classList.add('hidden');
        loginGate.style.display = 'flex'; // Show login gate

        setInputsLocked(true); // Lock the UI

        // Ensure websocket is closed
        if (websocket) {
            websocket.close();
            websocket = null;
        }
        // Use querySelector to be safe
        const systemMsg = document.querySelector("#messages .system");
        if(systemMsg) systemMsg.textContent = "Please sign in to establish a connection.";
    }
});

signInButton.addEventListener('click', () => {
    signInWithPopup(auth, provider).catch(error => {
        console.error("Sign-in error:", error);
        appendLog(`Sign-in failed: ${error.message}`, "system");
    });
});

signOutButton.addEventListener('click', () => {
    signOut(auth).catch(error => {
        console.error("Sign-out error:", error);
    });
});

// --- UI Control ---

function setInputsLocked(isLocked) {
    messageInput.disabled = isLocked;
    sendButton.disabled = isLocked;
    startAudioButton.disabled = isLocked;
    startVideoButton.disabled = isLocked;

    messageInput.placeholder = isLocked ? "Authenticate to transmit..." : "Transmit your message...";

}

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
function connectWebsocket(token) {
    if (!token) {
        console.error("Connection failed: No authentication token provided.");
        appendLog("Authentication error. Please sign in again.", "system");
        return;
    }
    if (websocket) websocket.close();

    const ws_protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    // **CRITICAL**: Properly encode the token parameter
    const encodedToken = encodeURIComponent(token);
    const ws_url = `${ws_protocol}//${window.location.host}/ws?token=${encodedToken}&is_audio=${is_audio_mode_active}`;
    
    console.log("Attempting to connect to secure WebSocket...");
    console.log("WebSocket URL (without token):", ws_url.replace(/token=[^&]+/, 'token=***'));
    
    websocket = new WebSocket(ws_url);

    websocket.onopen = () => {
        console.log("Secure WebSocket connection opened.");
        appendLog("Connection established. Ready for transmission.", "system");
         // Unlock the UI now that the connection is ready
        setInputsLocked(false); 
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
        //document.getElementById("sendButton").disabled = true;
         // Lock inputs while attempting to reconnect
        setInputsLocked(true); 
        showAgentThinking(false); // Ensure thinking indicator is hidden on close
        // Clear audio timers on connection close
        if (audioSilenceTimer) {
            clearTimeout(audioSilenceTimer);
            audioSilenceTimer = null;
        }
        isProcessingAudioResponse = false;
        // Reconnect with fresh token
        setTimeout(async () => {
            const user = auth.currentUser;
            if (user) {
                try {
                    const freshToken = await user.getIdToken(true);
                    connectWebsocket(freshToken);
                } catch (error) {
                    console.error("Error getting fresh token for reconnection:", error);
                    appendLog("Reconnection failed. Please refresh the page.", "system");
                }
            }
        }, 5000);
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
                logInteraction('text_message');
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
    logInteraction('audio_toggle');
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
            } else if (!websocket || (websocket.readyState !== WebSocket.OPEN && websocket.readyState !== WebSocket.CONNECTING)) {
            // Only connect if there isn't one already connecting
            const user = auth.currentUser;
            if (user) {
                const token = await user.getIdToken();
                connectWebsocket(token);
            }
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
    logInteraction('video_toggle');
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

// --- Analytics ---
function logInteraction(type) {
  if (typeof logEvent === 'function') {
    logEvent(analytics, 'interaction', {
      type: type,
      user_id: auth.currentUser ? auth.currentUser.uid : 'anonymous'
    });
  }
}
