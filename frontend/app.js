const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");
const result = document.getElementById("result");

const form = document.querySelector("form");

if (form) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
  });
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (!file) {
    preview.hidden = true;
    return;
  }

  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
});

analyzeButton.addEventListener("click", async (event) => {
  event.preventDefault();
  const image = imageInput.files[0];
  const crop = document.getElementById("crop").value;

  if (!image) {
    alert("Please select a tomato leaf image first.");
    return;
  }

  const formData = new FormData();
  formData.append("crop", crop);
  formData.append("image", image);

  analyzeButton.textContent = "Analyzing...";
  analyzeButton.disabled = true;

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction request failed.");
    }

    const organicItems = data.treatment.organic
      .map((item) => `<li>${item}</li>`)
      .join("");

    const chemicalItems = data.treatment.chemical
      .map((item) => `<li>${item}</li>`)
      .join("");

    result.innerHTML = `
      <h2>Screening Result</h2>
      <p><strong>Possible condition:</strong> ${data.prediction}</p>
      <p><strong>Confidence:</strong> ${data.confidence}%</p>
      <p><strong>Prototype severity:</strong> ${data.severity}</p>
      <p><strong>Advice:</strong> ${data.advisory}</p>
      <h3>Organic Management</h3>
      <ul class="treatment-list">${organicItems}</ul>
      <h3>Chemical Treatment for India / Uttar Pradesh</h3>
      <ul class="treatment-list">${chemicalItems}</ul>
      <p class="notice">
        This is a screening tool, not a guaranteed diagnosis. Chemical active ingredients are examples only. Confirm the current product label, dose, protective equipment, and harvest interval with a Uttar Pradesh KVK or agriculture officer before use.
      </p>
    `;

    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (error) {
    result.innerHTML = `
      <h2>Could not analyze the image</h2>
      <p>${error.message}</p>
    `;
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } finally {
    analyzeButton.textContent = "Analyze Image";
    analyzeButton.disabled = false;
  }
});
const historyButton = document.getElementById("historyButton");
const history = document.getElementById("history");
const historyContent = document.getElementById("historyContent");

historyButton.addEventListener("click", async () => {
  try {
    const response = await fetch("/reports");
    const reports = await response.json();

    if (reports.length === 0) {
      historyContent.innerHTML = "<p>No reports have been saved yet.</p>";
    } else {
      historyContent.innerHTML = reports.map((report) => `
        <article class="report-item">
          <p><strong>${report.prediction}</strong> - ${report.confidence}% confidence</p>
          <p>Crop: ${report.crop} | Severity: ${report.severity}</p>
          <p>${report.advisory}</p>
          <small>${new Date(report.created_at).toLocaleString()}</small>
        </article>
      `).join("");
    }

    history.hidden = false;
    history.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    historyContent.innerHTML = "<p>Could not load report history.</p>";
    history.hidden = false;
  }
});
