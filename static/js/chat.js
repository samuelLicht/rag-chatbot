const chatWidget = document.getElementById("chatWidget");
const chatLauncher = document.getElementById("chatLauncher");
const minimizeChat = document.getElementById("minimizeChat");
const clearChat = document.getElementById("clearChat");
const chatMessages = document.getElementById("chatMessages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");

const initialHTML = chatMessages.innerHTML;
const STORAGE_KEY = "colombiaComparteChatMinimized";

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHTML(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function renderBotMessage(text) {
    marked.use({ breaks: true, gfm: true });
    return marked.parse(text || "");
}

function addMessage(text, sender = "bot") {
    const row = document.createElement("div");
    row.className = `message-row ${sender === "user" ? "user-row" : "bot-row"}`;

    if (sender === "bot") {
        const avatar = document.createElement("div");
        avatar.className = "mini-avatar";
        avatar.setAttribute("aria-hidden", "true");
        const avatarImg = document.createElement("img");
        avatarImg.src = "/static/img/image.png";
        avatarImg.alt = "Latinoamérica Comparte";
        avatar.appendChild(avatarImg);
        row.appendChild(avatar);
    }

    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${sender === "user" ? "user-bubble" : "bot-bubble"}`;

    if (sender === "bot") {
        bubble.innerHTML = renderBotMessage(text);
    } else {
        bubble.textContent = text;
    }

    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();

    return row;
}

function addTyping() {
    const oldTyping = document.getElementById("typingIndicator");
    if (oldTyping) oldTyping.remove();

    const row = document.createElement("div");
    row.className = "message-row bot-row";
    row.id = "typingIndicator";

    const avatar = document.createElement("div");
    avatar.className = "mini-avatar";
    avatar.setAttribute("aria-hidden", "true");
    const typingImg = document.createElement("img");
    typingImg.src = "/static/img/logoLatin.png";
    typingImg.alt = "Latinoamérica Comparte";
    avatar.appendChild(typingImg);

    const bubble = document.createElement("div");
    bubble.className = "message-bubble bot-bubble";
    bubble.innerHTML = `<div class="typing" aria-label="El asistente está escribiendo"><span></span><span></span><span></span></div>`;
    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();
}

function removeTyping() {
    const typing = document.getElementById("typingIndicator");
    if (typing) typing.remove();
}

function setLoading(isLoading) {
    messageInput.disabled = isLoading;
    chatForm.querySelector("button").disabled = isLoading;
}

function minimizeWidget() {
    chatWidget.classList.add("minimized");
    chatLauncher.classList.remove("hidden");
    localStorage.setItem(STORAGE_KEY, "true");
}

function openWidget() {
    chatWidget.classList.remove("minimized");
    chatLauncher.classList.add("hidden");
    localStorage.setItem(STORAGE_KEY, "false");
    setTimeout(() => messageInput.focus(), 220);
}

async function sendMessage(message) {
    const cleanMessage = message.trim();
    if (!cleanMessage) return;

    if (chatWidget.classList.contains("minimized")) {
        openWidget();
    }

    addMessage(cleanMessage, "user");
    messageInput.value = "";
    setLoading(true);
    addTyping();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: cleanMessage })
        });

        const data = await response.json();
        removeTyping();

        if (!response.ok) {
            addMessage(data.answer || "Ocurrió un error al procesar la pregunta.", "bot");
        } else {
            addMessage(data.answer || "No recibí respuesta del servidor.", "bot");
        }
    } catch (error) {
        removeTyping();
        addMessage("No pude conectarme con el backend. Verifica que Flask esté corriendo.", "bot");
        console.error(error);
    } finally {
        setLoading(false);
        messageInput.focus();
    }
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(messageInput.value);
});

chatMessages.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-question]");
    if (!button) return;
    sendMessage(button.dataset.question);
});

minimizeChat.addEventListener("click", minimizeWidget);
chatLauncher.addEventListener("click", openWidget);

clearChat.addEventListener("click", () => {
    chatMessages.innerHTML = initialHTML;
    messageInput.value = "";
    messageInput.focus();
    scrollToBottom();
});

function restoreWidgetState() {
    const minimized = localStorage.getItem(STORAGE_KEY) === "true";

    if (minimized) {
        chatWidget.classList.add("minimized");
        chatLauncher.classList.remove("hidden");
    } else {
        chatWidget.classList.remove("minimized");
        chatLauncher.classList.add("hidden");
    }
}

restoreWidgetState();
scrollToBottom();
