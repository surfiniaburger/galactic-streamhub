// static/js/firebase-chat.js

import {
    collection,
    query,
    orderBy,
    limit,
    onSnapshot,
    addDoc
} from "https://www.gstatic.com/firebasejs/11.9.0/firebase-firestore.js";
import {
    ref,
    onValue,
    set as dbSet
} from "https://www.gstatic.com/firebasejs/11.9.0/firebase-database.js";

// Get services from the global window object initialized in index.html
const firestore = window.firebaseServices.firestore;
const realtimeDB = window.firebaseServices.realtimeDB;
const auth = window.firebaseServices.auth;

const messagesDiv = document.getElementById("messages");
const agentThinkingIndicator = document.getElementById("agent-thinking-indicator");

// --- Firestore Real-time Chat ---
const messagesRef = collection(firestore, "chats");
const chatQuery = query(messagesRef, orderBy("timestamp", "desc"), limit(50));

let isFirstLoad = true;
onSnapshot(chatQuery, (snapshot) => {
    if (isFirstLoad) {
        messagesDiv.innerHTML = ''; // Clear initial "Welcome" message
        isFirstLoad = false;
    }
    snapshot.docChanges().reverse().forEach((change) => {
        if (change.type === "added") {
            const msgData = change.doc.data();
            appendChatMessage(msgData);
        }
    });
    // Ensure the "thinking" indicator is always at the bottom if visible
    if (!agentThinkingIndicator.classList.contains('hidden')) {
        messagesDiv.appendChild(agentThinkingIndicator);
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
});

export async function sendChatMessage(text) {
    if (!auth.currentUser) {
        console.error("User not authenticated!");
        return;
    }
    try {
        await addDoc(messagesRef, {
            text: text,
            uid: auth.currentUser.uid,
            timestamp: new Date()
        });
    } catch (e) {
        console.error("Error sending message: ", e);
    }
}

function appendChatMessage(msg) {
    const isUser = msg.uid === auth.currentUser?.uid;
    const messageClass = isUser ? 'user' : 'remote';
    
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", messageClass);
    
    // Basic sanitization and URL replacement
    const sanitizedText = msg.text.replace(/</g, "<").replace(/>/g, ">");
    const imageUrlRegex = /(\/static\/charts\/[a-zA-Z0-9_-]+\.png)/g;
    messageDiv.innerHTML = sanitizedText.replace(
        imageUrlRegex, 
        '<br><img src="$1" alt="Generated Chart" style="max-width: 100%; display: block; margin: 10px 0; border: 1px solid #ccc; border-radius: 8px;"><br>'
    );

    // Insert message before the thinking indicator
    messagesDiv.insertBefore(messageDiv, agentThinkingIndicator);
}


// --- Realtime Database for Agent "Thinking" Status ---
const agentStatusRef = ref(realtimeDB, 'agentStatus/isThinking');

onValue(agentStatusRef, (snapshot) => {
    const isThinking = snapshot.val();
    if (isThinking) {
        agentThinkingIndicator.classList.remove("hidden");
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } else {
        agentThinkingIndicator.classList.add("hidden");
    }
});

export function setAgentThinking(isThinking) {
    dbSet(agentStatusRef, isThinking);
}