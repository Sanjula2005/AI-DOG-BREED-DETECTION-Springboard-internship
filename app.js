
const API = "http://127.0.0.1:5000";

window.onload = loadHistory;

// ==================== FIX: DEFINE selectedImage ====================
let selectedImage = null;

// ==================== LOGIN ====================
document.getElementById("loginForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();

    let username = document.getElementById("login-username").value;
    let password = document.getElementById("login-password").value;

    let res = await fetch(`${API}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
    });

    let data = await res.json();

    if (res.status !== 200) {
        document.getElementById("login-error").classList.remove("d-none");
        document.getElementById("login-error").innerText = data.error;
        return;
    }

    localStorage.setItem("token", data.token);
    window.location.href = "dashboard.html";
});

// ==================== REGISTER ====================
document.getElementById("registerForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();

    let username = document.getElementById("reg-username").value;
    let password = document.getElementById("reg-password").value;

    let res = await fetch(`${API}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
    });

    let data = await res.json();

    if (res.status !== 201) {
        document.getElementById("register-error").classList.remove("d-none");
        document.getElementById("register-error").innerText = data.error;
        return;
    }

    alert("Registration successful");
    window.location.href = "index.html";
});

// ==================== LOAD HISTORY ====================
async function loadHistory() {
    let res = await fetch(`${API}/history`, {
        headers: { Authorization: localStorage.getItem("token") }
    });

    let history = await res.json();

    let list = document.getElementById("history-list");
    if (!list) return;

    list.innerHTML = "";

    history.forEach(h => {
        let li = document.createElement("li");
        li.className = "list-group-item";
        li.innerHTML = `
            <strong>${h.breed}</strong> - ${h.confidence}% 
            <br>
            <small class="text-muted">${new Date(h.timestamp).toLocaleString()}</small>
        `;
        list.appendChild(li);
    });
}


// ==================== IMAGE UPLOAD PREVIEW ====================
let imageInput = document.getElementById("imageInput");
let dropArea = document.getElementById("drop-area");

dropArea.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    previewImage(file);
    selectedImage = file;
});

function previewImage(file) {
    const preview = document.getElementById("uploadPreview");
    const text = document.getElementById("uploadText");

    preview.classList.remove("d-none");
    text.style.display = "none";

    preview.src = URL.createObjectURL(file);
}

// ==================== PREDICT ====================
async function predictImage() {
    if (!selectedImage) {
        alert("Upload an image first!");
        return;
    }

    document.getElementById("previewImage").src = URL.createObjectURL(selectedImage);

    let formData = new FormData();
    formData.append("file", selectedImage);

    let btn = document.getElementById("predictBtn");
    btn.innerHTML = "Predicting...";
    btn.disabled = true;

    let res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { Authorization: localStorage.getItem("token") },
        body: formData
    });

    btn.innerHTML = "Predict";
    btn.disabled = false;

    let data = await res.json();
    if (res.status !== 200) return alert(data.error);

    let breed = data.predicted.breed;
    let confidence = data.predicted.confidence;
    window.lastPredictedBreed = breed;


    document.getElementById("predBreed").innerText = breed;
    document.getElementById("predConfidence").innerText = `Confidence: ${confidence}%`;

    // Save to history
    let history = JSON.parse(localStorage.getItem("history") || "[]");
    history.push(data.predicted);
    localStorage.setItem("history", JSON.stringify(history));

    loadHistory();

    new bootstrap.Modal(document.getElementById("predictionModal")).show();
}

// ==================== MORE INFO ====================
function goToBreedPage() {
    const breed = document.getElementById("predBreed").innerText;
    localStorage.setItem("selectedBreed", breed);

    // Force close modal using DOM API
    const modalEl = document.getElementById("predictionModal");
    modalEl.classList.remove("show");
    modalEl.style.display = "none";

    document.body.classList.remove("modal-open");
    document.querySelectorAll(".modal-backdrop").forEach(b => b.remove());

    // HARD redirect (cannot be blocked)
    setTimeout(() => {
        window.location.assign("breed.html");
    }, 100);
}




async function getBreedInfo() {
    let breed = document.getElementById("predBreed").innerText;

    let res = await fetch(`${API}/breed-info?breed=${breed}`, {
        headers: { Authorization: localStorage.getItem("token") }
    });

    let data = await res.json();

    // Build a clean formatted list
    let html = `
        <div style="text-align:left; font-size:16px; line-height:1.6;">
            <h3 style="text-align:center; margin-bottom:10px;">${data.Breed}</h3>
            <hr>
            ${Object.entries(data).map(([key, value]) => {
                if (key === "Breed") return "";
                return `
                    <div style="margin-bottom:8px;">
                        <strong>${key.replace(/_/g, " ")}:</strong>
                        <span>${value}</span>
                    </div>
                `;
            }).join("")}
        </div>
    `;

    Swal.fire({
        title: `${breed} Info`,
        html: html,
        width: 700,
        confirmButtonColor: "#3085d6",
        scrollbarPadding: false
    });
}

function logout() {
    localStorage.clear();
    window.location.href = "index.html";
}
