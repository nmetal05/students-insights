/**
 * EduPredict - Core Application Logic
 * Pure Vanilla JavaScript implementation of ML classification
 */

// --- Model Configuration (Extracted from Jupyter Notebook) ---
const MODEL_CONFIG = {
    scaler: {
        mean: [9.883191153685994, 80.18119506722138, 5.500453551210817],
        scale: [5.752314893222216, 11.571203423371777, 2.580834764341034]
    },
    logisticRegression: {
        coef: [
            [10.295743872225227, 3.712924665223109, 3.186329644632605],
            [4.657751685774865, 1.7778408988604228, 1.50793648322333],
            [-0.7428029301871368, -0.04259141051133599, -0.13478669098875173],
            [-5.977340454601074, -1.813516573309296, -1.7106684916485877],
            [-8.233352173211955, -3.6346575802628993, -2.848810945218603]
        ],
        intercept: [5.648461965455058, 7.025443835324328, 3.6505494073918396, -4.08065977737378, -12.243795430797299],
        classes: ["A", "B", "C", "D", "F"]
    }
};

const GRADE_INFO = {
    "A": { heading: "Exceptional Excellence", color: "#22c55e", message: "The model predicts high academic standing with major honors likely." },
    "B": { heading: "Strong Standing", color: "#38bdf8", message: "Superior performance detected. The student is well above the required threshold." },
    "C": { heading: "Standard Competency", color: "#eab308", message: "Satisfactory progress. Maintaining current habits will lead to a standard passing grade." },
    "D": { heading: "Improvement Required", color: "#f97316", message: "Marginal standing. Additional support or increased study hours may be necessary." },
    "F": { heading: "Critical Intervention", color: "#ef4444", message: "High-risk alert. Immediate intervention is advised to improve academic outcomes." }
};

// --- DOM Elements ---
const form = document.getElementById('predictionForm');
const resultSection = document.getElementById('resultSection');
const mainCard = document.querySelector('.main-card');
const predictedGradeEl = document.getElementById('predictedGrade');
const resultHeadingEl = document.getElementById('resultHeading');
const resultMessageEl = document.getElementById('resultMessage');
const resetBtn = document.getElementById('resetBtn');
const submitBtn = document.getElementById('submitBtn');

// --- ML Predictor Function ---
function predictGrade(features) {
    // 1. Feature Scaling (Z-score normalization)
    const scaledFeatures = features.map((x, i) => {
        return (x - MODEL_CONFIG.scaler.mean[i]) / MODEL_CONFIG.scaler.scale[i];
    });

    // 2. Logistic Regression (One-vs-Rest / Multinomial Score)
    const { coef, intercept, classes } = MODEL_CONFIG.logisticRegression;
    let scores = [];

    for (let c = 0; c < classes.length; c++) {
        let score = intercept[c];
        for (let i = 0; i < scaledFeatures.length; i++) {
            score += scaledFeatures[i] * coef[c][i];
        }
        scores.push(score);
    }

    // 3. Argmax to find the most probable class
    const maxScoreIndex = scores.indexOf(Math.max(...scores));
    return classes[maxScoreIndex];
}

// --- Event Handlers ---

form.addEventListener('submit', (e) => {
    e.preventDefault();

    // Disable button for a "processing" feel
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Processing Data...</span>';

    setTimeout(() => {
        const studyHours = parseFloat(document.getElementById('studyHours').value);
        const attendance = parseFloat(document.getElementById('attendance').value);
        const participation = parseFloat(document.getElementById('participation').value);

        const grade = predictGrade([studyHours, attendance, participation]);
        showResult(grade);

        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }, 600);
});

function showResult(grade) {
    const info = GRADE_INFO[grade];

    // Update content
    predictedGradeEl.textContent = grade;
    predictedGradeEl.parentElement.style.borderColor = info.color;
    predictedGradeEl.parentElement.style.boxShadow = `0 0 40px ${info.color}66`;
    resultHeadingEl.textContent = info.heading;
    resultHeadingEl.style.color = info.color;
    resultMessageEl.textContent = info.message;

    // UI Animations
    mainCard.style.display = 'none';
    resultSection.classList.remove('hidden');
    resultSection.style.display = 'block';

    // Trigger entrance animation
    requestAnimationFrame(() => {
        resultSection.style.opacity = '1';
        resultSection.style.transform = 'translateY(0)';
    });
}

resetBtn.addEventListener('click', () => {
    // Hide results
    resultSection.style.display = 'none';
    resultSection.style.opacity = '0';
    resultSection.style.transform = 'translateY(20px)';

    // Show form
    mainCard.style.display = 'block';
    form.reset();
});

// --- Aesthetic Micro-interactions ---
document.querySelectorAll('input').forEach(input => {
    input.addEventListener('focus', () => {
        input.parentElement.parentElement.style.transform = 'scale(1.02)';
        input.parentElement.parentElement.style.transition = 'transform 0.2s ease';
    });
    input.addEventListener('blur', () => {
        input.parentElement.parentElement.style.transform = 'scale(1)';
    });
});
