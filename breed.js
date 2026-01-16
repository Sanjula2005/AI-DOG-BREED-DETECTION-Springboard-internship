const API = "http://127.0.0.1:5000";
const breed = localStorage.getItem("selectedBreed");

if (!breed) {
    alert("No breed selected");
    window.location.href = "dashboard.html";
}

// console.log("DIET DATA:", data);


// ---------------- LOAD BREED INFO ----------------
async function loadBreedInfo() {
    const res = await fetch(`${API}/breed-info?breed=${breed}`, {
        headers: { Authorization: localStorage.getItem("token") }
    });

    const data = await res.json();

    let html = `<h3 class="fw-bold text-center mb-3">${data.Breed}</h3><hr>`;

    Object.entries(data).forEach(([key, value]) => {
        if (key !== "Breed") {
            html += `
                <p>
                    <strong>${key.replace(/_/g, " ")}:</strong> ${value}
                </p>
            `;
        }
    });

    document.getElementById("breedInfo").innerHTML = html;
}

loadBreedInfo();

// ---------------- LOAD DIET PLAN ----------------
async function loadDietPlan() {
    const stage = document.getElementById("lifeStage").value;
    if (!stage) return alert("Please select life stage");

    const resultDiv = document.getElementById("dietResult");

    resultDiv.innerHTML = `
        <div class="text-center text-muted">
            ⏳ Loading diet plan…
        </div>
    `;

    try {
        const res = await fetch(
            `${API}/diet-plan?stage=${stage}`,
            {
                headers: {
                    Authorization: localStorage.getItem("token")
                }
            }
        );

        const data = await res.json();
        console.log("DIET RESPONSE:", data);

        if (!res.ok) {
            resultDiv.innerHTML =
                `<div class="alert alert-danger">${data.error}</div>`;
            return;
        }

        resultDiv.innerHTML = `
            <h6 class="fw-bold mb-2">${stage} Diet Plan</h6>

            <ul class="list-group mb-3">
                <li class="list-group-item">
                    <strong>🍳 Breakfast:</strong><br>${data.breakfast}
                </li>
                <li class="list-group-item">
                    <strong>🍛 Lunch:</strong><br>${data.lunch}
                </li>
                <li class="list-group-item">
                    <strong>🍪 Snack:</strong><br>${data.snack}
                </li>
                <li class="list-group-item">
                    <strong>🍖 Dinner:</strong><br>${data.dinner}
                </li>
            </ul>

            <div class="alert alert-warning">
                <strong>⚠️ Notes:</strong><br>${data.notes}
            </div>
        `;

    } catch (err) {
        console.error(err);
        resultDiv.innerHTML =
            `<div class="alert alert-danger">Failed to load diet plan</div>`;
    }
}

// ---------------- NAV ----------------
function goHome() {
    window.location.href = "dashboard.html";
}
