// chatbot.js (replace existing chatbot code with this)

// Floating Chat Button
const botBtn = document.createElement("div");
botBtn.id = "chatbot-btn";
botBtn.innerHTML = "\u{1F4AC}\u{1F43E}";  // 💬

document.body.appendChild(botBtn);

// Chat Window
const botWindow = document.createElement("div");
botWindow.id = "chatbot-window";
botWindow.innerHTML = `
  <div class="chat-header bg-primary text-white p-2">
      <strong>Dog Chatbot</strong>
  </div>

  <div id="chat-body" class="chat-body p-2"></div>

  <div id="chat-controls" class="chat-controls p-2">
    <textarea id="chat-msg" class="form-control" rows="2" placeholder="Ask something..."></textarea>
    <button class="btn btn-primary w-100 mt-2" id="chat-send-btn">Send</button>
  </div>
`;
document.body.appendChild(botWindow);

// Ensure hidden by default (class-based)
botWindow.classList.remove("open"); // safe no-op if class missing

// Toggle Chat Window (use class toggle so CSS controls display)
botBtn.onclick = () => {
    botWindow.classList.toggle("open");
    // focus textarea when opened
    if (botWindow.classList.contains("open")) {
        setTimeout(() => document.getElementById("chat-msg").focus(), 100);
    }
};

// Format AI Response for UI
function formatBotMessage(text) {
    if (!text) return "";
    let formatted = text;

    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>"); // Bold
    formatted = formatted.replace(/\n{2,}/g, "<br><br>");                  // double line breaks -> paragraph spacing
    formatted = formatted.replace(/\n/g, "<br>");                          // single line -> <br>
    formatted = formatted.replace(/•/g, "<br>•");                           // Bullet Points
    formatted = formatted.replace(/(\n|^)(\d+)\.\s+/g, "<br><strong>$2.</strong> "); // Numbering

    return formatted;
}

// SEND CHAT MESSAGE
async function sendChat() {
    const msgEl = document.getElementById("chat-msg");
    let msg = msgEl.value || "";
    let body = document.getElementById("chat-body");

    if (!msg.trim()) return;

    // USER MESSAGE BUBBLE
    body.innerHTML += `
        <div class="msg-bubble user-msg">
            <b>You:</b> ${msg.replace(/</g, "&lt;").replace(/>/g, "&gt;")}
        </div>
    `;
    body.scrollTop = body.scrollHeight;

    // Prepare chat request
    let form = new FormData();
    form.append("breed", window.lastPredictedBreed || "dog");
    form.append("user_input", msg);

    try {
        const res = await fetch(`${API}/chat`, {
            method: "POST",
            headers: { Authorization: localStorage.getItem("token") },
            body: form
        });

        const data = await res.json();

        // BOT MESSAGE BUBBLE
        body.innerHTML += `
            <div class="msg-bubble bot-msg">
                <b>Bot:</b><br>
                ${formatBotMessage(data.reply)}
            </div>
        `;
    } catch (err) {
        body.innerHTML += `
            <div class="msg-bubble bot-msg">
                <b>Bot:</b><br>
                Something went wrong. Please try again.
            </div>
        `;
        console.error("chat error:", err);
    }

    msgEl.value = "";
    body.scrollTop = body.scrollHeight;
}

// Wire the send button and enter key
document.addEventListener("click", (e) => {
    if (e.target && e.target.id === "chat-send-btn") sendChat();
});

// Enter to send (Shift+Enter for newline)
document.addEventListener("keydown", (e) => {
    const el = document.activeElement;
    if (el && el.id === "chat-msg") {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChat();
        }
    }
});
