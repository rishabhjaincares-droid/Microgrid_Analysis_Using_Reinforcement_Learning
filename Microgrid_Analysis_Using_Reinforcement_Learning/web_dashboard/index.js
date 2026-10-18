/* ==========================================
   Smart Microgrid RL Dashboard — JavaScript
   ========================================== */

// ===== Scenario Data =====
const scenarioData = {
    november: {
        rl: { bill: 649.68, cycles: 22.65, gridImport: 149.34, exported: 130.25, peakImport: 10.76, gridCharge: 6.93 },
        baseline: { bill: 745.11, cycles: 17.70, gridImport: 168.21, exported: 150.84, peakImport: 23.77, gridCharge: 2.00 },
        label: "November 2024 (Full Month)"
    },
    "mid-nov": {
        rl: { bill: 177.03, cycles: 5.40, gridImport: 33.46, exported: 18.91, peakImport: 2.16, gridCharge: 1.03 },
        baseline: { bill: 204.70, cycles: 4.14, gridImport: 38.80, exported: 23.95, peakImport: 4.99, gridCharge: 0.00 },
        label: "Mid-November Peak Week"
    },
    "mid-dec": {
        rl: { bill: 172.33, cycles: 5.40, gridImport: 34.86, exported: 23.15, peakImport: 2.33, gridCharge: 3.06 },
        baseline: { bill: 199.84, cycles: 3.97, gridImport: 39.84, exported: 28.31, peakImport: 6.01, gridCharge: 1.00 },
        label: "Mid-December Peak Week"
    }
};

// ===== Solar simulation profile (typical day in Hosur) =====
const solarProfile = [0, 0, 0, 0, 0, 0, 0.10, 0.66, 1.30, 1.91, 2.29, 2.38, 2.27, 1.96, 1.51, 0.93, 0.39, 0.02, 0, 0, 0, 0, 0, 0];
const loadProfile = [0.45, 0.42, 0.40, 0.38, 0.38, 0.40, 0.71, 0.93, 0.70, 0.52, 0.48, 0.48, 0.50, 0.50, 0.48, 0.48, 0.50, 0.65, 1.20, 1.18, 1.15, 1.00, 0.55, 0.48];
const socProfile = [0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.38, 0.42, 0.52, 0.64, 0.75, 0.82, 0.88, 0.92, 0.93, 0.90, 0.82, 0.72, 0.62, 0.55, 0.52, 0.50];

const tariffRates = {
    normal: 7.00,
    solar: 5.60,
    peak: 8.40
};

function getTariffInfo(hour) {
    if (hour >= 9 && hour <= 16) return { period: "Solar Hour", rate: tariffRates.solar, color: "var(--accent-500)" };
    if (hour >= 18 && hour <= 21) return { period: "Peak Hour", rate: tariffRates.peak, color: "var(--secondary-500)" };
    return { period: "Normal", rate: tariffRates.normal, color: "var(--text-500)" };
}

const rlActions = ["idle", "idle", "idle", "idle", "idle", "idle", "idle", "discharge_low", "charge_low", "charge_high", "charge_high", "charge_high", "charge_high", "charge_high", "charge_low", "charge_low", "idle", "idle", "discharge_high", "discharge_high", "discharge_low", "discharge_low", "idle", "idle"];

// ===== Theme Toggle =====
const themeToggle = document.getElementById("themeToggle");
function getEffectiveTheme() {
    const current = document.documentElement.getAttribute("data-theme");
    if (current) return current;
    const saved = localStorage.getItem("theme");
    if (saved) return saved;
    return "dark"; // Default to the sleek dark theme from Realtime Colors
}

// Initialize theme immediately
const initialTheme = localStorage.getItem("theme") || "dark";
document.documentElement.setAttribute("data-theme", initialTheme);

if (themeToggle) {
    themeToggle.addEventListener("click", () => {
        const current = getEffectiveTheme();
        const next = current === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        localStorage.setItem("theme", next);
        initParticles();
    });
}

// ===== Navbar Scroll =====
const navbar = document.getElementById("navbar");
window.addEventListener("scroll", () => {
    navbar.classList.toggle("scrolled", window.scrollY > 20);
    updateActiveNavLink();
});

function updateActiveNavLink() {
    const sections = document.querySelectorAll("section[id]");
    const navLinks = document.querySelectorAll(".nav-link");
    let current = "";

    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        if (window.scrollY >= sectionTop) {
            current = section.getAttribute("id");
        }
    });

    navLinks.forEach(link => {
        link.classList.remove("active");
        if (link.getAttribute("href") === "#" + current) {
            link.classList.add("active");
        }
    });
}

// ===== Animated Counters =====
function animateCounters() {
    const counters = document.querySelectorAll(".stat-value[data-count]");
    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute("data-count"));
        const isFloat = target % 1 !== 0;
        const duration = 2000;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = target * easeOut;

            counter.textContent = isFloat ? current.toFixed(1) : Math.floor(current).toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    });
}

// ===== Scroll Reveal Observer =====
const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px"
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

document.querySelectorAll(".arch-card, .result-highlight-card, .scenario-card, .config-card, .mdp-card, .metric-card, .chart-card").forEach(el => {
    observer.observe(el);
});

// Hero counter animation on load
const heroObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateCounters();
            heroObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.3 });
heroObserver.observe(document.querySelector(".hero-stats"));

// ===== Scenario Tab Switching =====
const scenarioTabs = document.querySelectorAll(".scenario-tab");
scenarioTabs.forEach(tab => {
    tab.addEventListener("click", () => {
        scenarioTabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        updateResults(tab.getAttribute("data-scenario"));
    });
});

function updateResults(scenarioKey) {
    const data = scenarioData[scenarioKey];
    if (!data) return;

    const billSavings = data.baseline.bill - data.rl.bill;
    const billSavingsPercent = ((billSavings / data.baseline.bill) * 100).toFixed(1);
    const peakCut = data.baseline.peakImport - data.rl.peakImport;
    const peakCutPercent = ((peakCut / data.baseline.peakImport) * 100).toFixed(1);
    const gridCut = data.baseline.gridImport - data.rl.gridImport;
    const gridCutPercent = ((gridCut / data.baseline.gridImport) * 100).toFixed(1);

    document.getElementById("billSavings").textContent = `₹${billSavings.toFixed(2)}`;
    document.getElementById("billSavingsPercent").textContent = `▼ ${billSavingsPercent}% vs baseline`;
    document.getElementById("peakCut").textContent = `${peakCut.toFixed(2)} kWh`;
    document.getElementById("peakCutPercent").textContent = `▼ ${peakCutPercent}% less peak grid`;
    document.getElementById("gridImportCut").textContent = `${gridCut.toFixed(2)} kWh`;
    document.getElementById("gridImportPercent").textContent = `▼ ${gridCutPercent}% lower import`;

    // Update comparison table
    const tbody = document.getElementById("comparisonBody");
    tbody.innerHTML = `
        <tr class="rl-row">
            <td><span class="rl-badge">⚡ RL Agent</span></td>
            <td class="winner-cell">₹${data.rl.bill.toFixed(2)}</td>
            <td>${data.rl.cycles.toFixed(2)}</td>
            <td class="winner-cell">${data.rl.gridImport.toFixed(2)}</td>
            <td>${data.rl.exported.toFixed(2)}</td>
            <td class="winner-cell">${data.rl.peakImport.toFixed(2)}</td>
            <td>${data.rl.gridCharge.toFixed(2)}</td>
        </tr>
        <tr>
            <td><span class="baseline-badge">📋 Rule-Based</span></td>
            <td>₹${data.baseline.bill.toFixed(2)}</td>
            <td>${data.baseline.cycles.toFixed(2)}</td>
            <td>${data.baseline.gridImport.toFixed(2)}</td>
            <td>${data.baseline.exported.toFixed(2)}</td>
            <td>${data.baseline.peakImport.toFixed(2)}</td>
            <td>${data.baseline.gridCharge.toFixed(2)}</td>
        </tr>
    `;

    // Update charts
    updateCharts(data);
}

function updateCharts(data) {
    renderBarChart("billChart", "Bill Cost (₹)", [
        { label: "RL Agent", value: data.rl.bill, isRL: true },
        { label: "Rule-Based", value: data.baseline.bill, isRL: false }
    ]);

    renderBarChart("peakChart", "Peak Import (kWh)", [
        { label: "RL Agent", value: data.rl.peakImport, isRL: true },
        { label: "Rule-Based", value: data.baseline.peakImport, isRL: false }
    ]);

    renderBarChart("gridChart", "Grid Import (kWh)", [
        { label: "RL Agent", value: data.rl.gridImport, isRL: true },
        { label: "Rule-Based", value: data.baseline.gridImport, isRL: false }
    ]);

    renderBarChart("cycleChart", "Battery Cycles", [
        { label: "RL Agent", value: data.rl.cycles, isRL: true },
        { label: "Rule-Based", value: data.baseline.cycles, isRL: false }
    ]);
}

function renderBarChart(containerId, label, items) {
    const container = document.getElementById(containerId);
    const maxValue = Math.max(...items.map(i => i.value));

    container.innerHTML = items.map(item => {
        const percent = Math.max((item.value / maxValue) * 100, 8);
        const barClass = item.isRL ? "rl-bar" : "baseline-bar";
        const displayValue = item.value >= 10 ? item.value.toFixed(1) : item.value.toFixed(2);
        const showValueInside = percent > 25;

        return `
            <div class="bar-row">
                <span class="bar-label">${item.label}</span>
                <div class="bar-track">
                    <div class="bar-fill ${barClass}" style="width: 0%;" data-width="${percent}">
                        ${showValueInside ? `<span class="bar-value">${displayValue}</span>` : ""}
                    </div>
                </div>
                ${!showValueInside ? `<span class="bar-value-outside">${displayValue}</span>` : ""}
            </div>
        `;
    }).join("");

    // Animate bars
    requestAnimationFrame(() => {
        container.querySelectorAll(".bar-fill").forEach(bar => {
            bar.style.width = bar.getAttribute("data-width") + "%";
        });
    });
}

// ===== Hour Slider / Energy Flow =====
const hourSlider = document.getElementById("hourSlider");

hourSlider.addEventListener("input", () => {
    updateEnergyFlow(parseInt(hourSlider.value));
});

function updateEnergyFlow(hour) {
    const solar = solarProfile[hour];
    const load = loadProfile[hour];
    const soc = socProfile[hour];
    const tariff = getTariffInfo(hour);

    document.getElementById("currentHour").textContent = `${String(hour).padStart(2, "0")}:00`;
    document.getElementById("solarValue").textContent = `${solar.toFixed(2)} kW`;
    document.getElementById("batteryValue").textContent = `SOC: ${Math.round(soc * 100)}%`;
    document.getElementById("socFill").style.width = `${soc * 100}%`;
    document.getElementById("gridValue").textContent = `₹${tariff.rate.toFixed(2)}/kWh`;
    document.getElementById("loadValue").textContent = `${load.toFixed(2)} kW`;

    document.getElementById("tariffPeriod").textContent = tariff.period;
    document.getElementById("tariffPeriod").style.color = tariff.color;
    document.getElementById("importRate").textContent = `₹${tariff.rate.toFixed(2)}/kWh`;
    document.getElementById("rlAction").textContent = rlActions[hour];

    // Update SOC fill color based on level
    const socFill = document.getElementById("socFill");
    if (soc > 0.7) {
        socFill.style.background = "linear-gradient(90deg, #10b981, #059669)";
    } else if (soc > 0.35) {
        socFill.style.background = "linear-gradient(90deg, var(--secondary-500), var(--secondary-400))";
    } else {
        socFill.style.background = "linear-gradient(90deg, var(--accent-500), var(--accent-400))";
    }

    // Show/hide flow paths based on hour
    const solarToLoad = document.querySelector(".solar-to-load");
    const solarToBattery = document.querySelector(".solar-to-battery");
    const batteryToLoad = document.querySelector(".battery-to-load");
    const gridToLoad = document.querySelector(".grid-to-load");

    solarToLoad.style.opacity = solar > 0.1 ? "1" : "0.15";
    solarToLoad.style.strokeWidth = Math.min(solar / 2.5 * 4, 5) + "px";

    solarToBattery.style.opacity = (solar > load && soc < 0.9) ? "1" : "0.15";

    batteryToLoad.style.opacity = (hour >= 18 && hour <= 22 && soc > 0.3) ? "1" : "0.15";
    batteryToLoad.style.strokeWidth = (hour >= 18 && hour <= 22) ? "3px" : "1.5px";

    gridToLoad.style.opacity = (load > solar && soc < 0.4) || (hour >= 0 && hour <= 5) ? "1" : "0.15";
}

// ===== Particle Background =====
function initParticles() {
    const canvas = document.getElementById("particleCanvas");
    const ctx = canvas.getContext("2d");
    let particles = [];
    let animationId;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener("resize", resize);

    const isDark = getEffectiveTheme() === "dark";
    const particleCount = Math.min(60, Math.floor(window.innerWidth / 20));

    particles = Array.from({ length: particleCount }, () => ({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 2 + 1,
        opacity: Math.random() * 0.3 + 0.05
    }));

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const color = isDark ? "119, 131, 243" : "90, 103, 220";

        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${color}, ${p.opacity})`;
            ctx.fill();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(${color}, ${0.05 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        animationId = requestAnimationFrame(draw);
    }

    if (animationId) cancelAnimationFrame(animationId);
    draw();
}

// ===== Smooth Nav Link Scrolling =====
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const target = document.querySelector(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ===== Initialize =====
document.addEventListener("DOMContentLoaded", () => {
    initParticles();
    updateResults("november");
    updateEnergyFlow(12);
});
