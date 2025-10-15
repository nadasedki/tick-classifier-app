let file;
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const dropText = document.getElementById('dropText');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');

// Click opens file picker
dropzone.addEventListener('click', () => fileInput.click());

// File picker change
fileInput.addEventListener('change', e => {
    file = e.target.files[0];
    showPreview(file);
});

// Drag over effect
dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', e => {
    dropzone.classList.remove('dragover');
});

// Drop event
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    file = e.dataTransfer.files[0];
    showPreview(file);
});

// Show image preview
function showPreview(file) {
    if (!file) return;
    let reader = new FileReader();
    reader.onload = function(e) {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
        dropText.style.display = 'none';
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Predict click
function sendPrediction(file) {
    if (!file) return;
    let formData = new FormData();
    formData.append("file", file);

    predictBtn.disabled = true;
    predictBtn.innerText = "Predicting...";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(async res => {
        const text = await res.text();
        try {
            return JSON.parse(text);
        } catch {
            throw new Error("Invalid JSON response: " + text);
        }   
    })
    .then(data => {
        predictBtn.disabled = false;
        predictBtn.innerText = "Predict";

        if (data.error) {
            resultDiv.innerHTML = `<span style="color:red">${data.error}</span>`;
        } else {
            if (data.species === "unknown" || data.sex === "unknown") {
                // Show only "unknown" values
                resultDiv.innerHTML = `<b>Species and Sex: "unknown"</b>`;
            } else {
                // Show top prediction
                resultDiv.innerHTML = `
                    <b>Species: Hyalomma </b> ${data.species} (${data.species_confidence}%)<br>
                    <b>Sex:</b> ${data.sex} (${data.sex_confidence}%)
                `;
                 /*
                // Optional: show full confidence breakdown
                let breakdownHTML = "<br><b>All species probabilities:</b><br>";
                for (const [k,v] of Object.entries(data.species_all)){
                    breakdownHTML += `${k}: ${v}%<br>`;
                }
                breakdownHTML += "<br><b>All sex probabilities:</b><br>";
                for (const [k,v] of Object.entries(data.sex_all)){
                    breakdownHTML += `${k}: ${v}%<br>`;
                }
                let breakdownHTML1 = "<br><b>All species probabilities:</b>";
                breakdownHTML1 += '<div class="probability-grid">';
                for (const [k, v] of Object.entries(data.species_all)) {
                    breakdownHTML1 += `<div class="probability-item">${k}: <b>${v}%</b></div>`;
                }
                breakdownHTML1 += "</div>";

                breakdownHTML1 += "<br><b>All sex probabilities:</b>";
                breakdownHTML1 += '<div class="probability-grid">';
                for (const [k, v] of Object.entries(data.sex_all)) {
                    breakdownHTML1 += `<div class="probability-item">${k}: <b>${v}%</b></div>`;
                }
                breakdownHTML1 += "</div>";
                */
                // Enhanced species & sex probabilities display
let breakdownHTML2 = `
  <div class="mt-4">
    <button onclick="document.getElementById('probSection').classList.toggle('hidden')" 
            class="mt-2 px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 font-medium">
       All Probabilities
    </button>

    <section id="probSection" class="mt-3 hidden">
      <h4 class="font-semibold mb-2">All Species Probabilities</h4>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
`;

for (const [species, confidence] of Object.entries(data.species_all)) {
  breakdownHTML2 += `
    <div class="flex flex-col">
      <span class="text-sm font-medium">${species} (${confidence}%)</span>
      <div class="w-full bg-gray-200 rounded-full h-3">
        <div class="h-3 rounded-full" style="width:${confidence}%; background:${getBarColor(confidence)};"></div>
      </div>
    </div>
  `;
}

breakdownHTML2 += `
      </div>

      <h4 class="font-semibold mt-4 mb-2">All Sex Probabilities</h4>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
`;

for (const [sex, confidence] of Object.entries(data.sex_all)) {
  breakdownHTML2 += `
    <div class="flex flex-col">
      <span class="text-sm font-medium">${sex} (${confidence}%)</span>
      <div class="w-full bg-gray-200 rounded-full h-3">
        <div class="h-3 rounded-full" style="width:${confidence}%; background:${getBarColor(confidence)};"></div>
      </div>
    </div>
  `;
}

breakdownHTML2 += `
      </div>
    </section>
  </div>
`;

// Append to resultDiv
resultDiv.innerHTML += breakdownHTML2;

            }
        }
        
        /*if (data.gradcam_url) {
            resultDiv.innerHTML += `<br><img src="${data.gradcam_url}" width="300">`;
        }*/
          if (data.gradcam_image) {
            resultDiv.innerHTML += `<br><img src="data:image/jpeg;base64,${data.gradcam_image}" width="300">`;
}
        resultDiv.style.display = 'block';
    })
    .catch(err => {
        predictBtn.disabled = false;
        predictBtn.innerText = "Predict";
        resultDiv.innerHTML = `<span style="color:red">Error: ${err}</span>`;
        resultDiv.style.display = 'block';
    });
};

// Auto-predict when file is selected
fileInput.addEventListener('change', e => {
    file = e.target.files[0];
    showPreview(file);
    sendPrediction(file);  // auto-predict
});

// Auto-predict when file is dropped
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    file = e.dataTransfer.files[0];
    showPreview(file);
    sendPrediction(file);  // auto-predict
});

// Keep Predict button for manual trigger
predictBtn.addEventListener('click', () => {
    sendPrediction(file);
});

function getBarColor(value) {
    if (value >= 80) return "#4caf50"; // green
    if (value >= 50) return "#ff9800"; // orange
    return "#f44336"; // red
}

// Function to switch sections
        function showSection(sectionId) {
            document.querySelectorAll('.section').forEach(sec => sec.classList.add('hidden'));
            document.getElementById(sectionId).classList.remove('hidden');
            if (sectionId === 'dashboard') {
                loadDashboard()}; 
            // Load predictions if switching to that section
            if (sectionId === 'predictions') {
                fetchPredictions();
            }
        }

        // Fetch saved predictions from backend
        async function fetchPredictions() {
            try {
                const res = await fetch('/all'); // You'll expose this in Flask
                const data = await res.json();

                const container = document.getElementById('predictionsList');
                container.innerHTML = '';

                data.forEach(pred => {
                    const card = document.createElement('div');
                    card.className = 'border rounded-xl shadow p-4 bg-gray-50';

                    card.innerHTML = `
                        <p><strong>Species:</strong> ${pred.species}</p>
                        <p><strong> species Confidence:</strong> ${(pred.species_confidence).toFixed(2)}%</p>
                        <p><strong>sex Confidence:</strong> ${(pred.sex_confidence).toFixed(2)}%</p>
                        <img src="/image/${pred.original_image_id}" class="mt-2 rounded-lg shadow w-full">
                        <img src="/image/${pred.gradcam_image_id}" class="mt-2 rounded-lg shadow w-full">
                    `;

                    container.appendChild(card);
                });
            } catch (err) {
                console.error("Error loading predictions:", err);
            }
        }
/*
async function loadDashboard() {
    try {
        const res = await fetch('/api/dashboard-data');
        const data = await res.json();

        // Species distribution pie chart
        const speciesCtx = document.getElementById('speciesChart').getContext('2d');
        new Chart(speciesCtx, {
            type: 'pie',
            data: {
                labels: Object.keys(data.species_count),
                datasets: [{
                    label: 'Species Count',
                    data: Object.values(data.species_count),
                    backgroundColor: [
                        '#4caf50', '#ff9800', '#f44336', '#2196f3', '#9c27b0', '#ffc107'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'right' },
                    title: { display: true, text: 'Species Distribution' }
                }
            }
        });

        // Species average confidence pie chart
        const confCtx = document.getElementById('speciesConfChart').getContext('2d');
        new Chart(confCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(data.species_avg_conf),
                datasets: [{
                    label: 'Avg Species Confidence (%)',
                    data: Object.values(data.species_avg_conf),
                    backgroundColor: [
                        '#4caf50', '#ff9800', '#f44336', '#2196f3', '#9c27b0', '#ffc107'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'right' },
                    title: { display: true, text: 'Average Species Confidence' }
                }
            }
        });

    } catch(err) {
        console.error("Failed to load dashboard data:", err);
    }
}
*/
let allDashboardData = null; // store full data for filtering

async function loadDashboard() {
    try {
        const res = await fetch('/api/dashboard-data');
        const data = await res.json();
        allDashboardData = data; // store globally for filters

        populateFilters(data);
        updateCharts(); // initial draw
        updateKPIs();
    } catch(err) {
        console.error("Failed to load dashboard data:", err);
    }
}

function populateFilters(data) {
    const speciesDropdown = document.getElementById('filterSpecies');
    speciesDropdown.innerHTML = '<option value="">All</option>';

    Object.keys(data.species_count).forEach(sp => {
        const option = document.createElement('option');
        option.value = sp;
        option.textContent = sp;
        speciesDropdown.appendChild(option);
    });

    // Attach change listeners
    speciesDropdown.addEventListener('change', updateCharts);
    document.getElementById('filterConfidence').addEventListener('input', e => {
        document.getElementById('filterConfidenceVal').textContent = e.target.value + '%';
        updateCharts();
    });
}

function updateCharts() {
    const selectedSpecies = document.getElementById('filterSpecies').value;
    const minConfidence = parseInt(document.getElementById('filterConfidence').value);

    // Filter data
    const species_count = {};
    const species_avg_conf = {};

    for (const [sp, count] of Object.entries(allDashboardData.species_count)) {
        if (selectedSpecies && sp !== selectedSpecies) continue;
        const avgConf = allDashboardData.species_avg_conf[sp];
        if (avgConf < minConfidence) continue;

        species_count[sp] = count;
        species_avg_conf[sp] = avgConf;
    }

    // Destroy previous charts if exist
    if (window.speciesChartInstance) window.speciesChartInstance.destroy();
    if (window.speciesConfChartInstance) window.speciesConfChartInstance.destroy();

    // Draw Species Distribution
    const speciesCtx = document.getElementById('speciesChart').getContext('2d');
    window.speciesChartInstance = new Chart(speciesCtx, {
        type: 'pie',
        data: {
            labels: Object.keys(species_count),
            datasets: [{
                label: 'Species Count',
                data: Object.values(species_count),
                backgroundColor: ['#4caf50', '#ff9800', '#f44336', '#2196f3', '#9c27b0', '#ffc107']
            }]
        },
        options: { responsive: true }
    });

    // Draw Average Confidence
    const confCtx = document.getElementById('speciesConfChart').getContext('2d');
    window.speciesConfChartInstance = new Chart(confCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(species_avg_conf),
            datasets: [{
                label: 'Avg Species Confidence (%)',
                data: Object.values(species_avg_conf),
                backgroundColor: ['#4caf50', '#ff9800', '#f44336', '#2196f3', '#9c27b0', '#ffc107']
            }]
        },
        options: { responsive: true, scales: { y: { max: 100 } } }
    });
}

function updateKPIs() {
    const total = Object.values(allDashboardData.species_count).reduce((a,b) => a+b, 0);
    document.getElementById('kpi-total').textContent = total;

    const mostCommon = Object.entries(allDashboardData.species_count)
        .sort((a,b) => b[1]-a[1])[0];
    document.getElementById('kpi-species').textContent = mostCommon ? mostCommon[0] : '-';

    const avgConf = Object.values(allDashboardData.species_avg_conf);
    const overallAvg = avgConf.length ? (avgConf.reduce((a,b) => a+b,0)/avgConf.length).toFixed(1) : 0;
    document.getElementById('kpi-confidence').textContent = overallAvg + '%';
}
