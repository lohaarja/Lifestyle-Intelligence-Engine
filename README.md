An interactive deep learning application that analyzes lifestyle patterns and suggests improvements based on user goals.
The system uses a neural autoencoder to learn common lifestyle patterns and identify how different behaviors affect alignment with personal goals like better sleep, reduced stress, or improved fitness.
Built with **PyTorch + Streamlit + Plotly**.
---

## 🚀 Features
• Analyze lifestyle habits interactively  
• Detect how "unusual" a lifestyle pattern is  
• Visualize patterns using radar charts  
• Measure alignment with health goals  
• Generate actionable improvement suggestions  
• Simulate behavior changes and estimate impact  

---

## 🧠 How It Works
The system learns patterns from lifestyle data using an **Autoencoder neural network**.
Steps:
1. User inputs lifestyle metrics (sleep, stress, activity, screen time, etc.)
2. Inputs are normalized using a trained scaler
3. Data passes through the trained **Lifestyle Autoencoder**
4. Reconstruction error measures how unusual the pattern is
5. Latent embeddings represent lifestyle structure
6. Distance to goal embeddings determines **goal alignment**
7. Simulated adjustments estimate which habits give the biggest improvement

---

## 📊 Inputs
The model uses 14 lifestyle features including:
- Sleep Duration
- Sleep Quality
- Physical Activity Level
- Stress Level
- BMI Category
- Workout Duration
- Calories Burned
- Workout Frequency
- Body Fat Percentage
- BMI
- Daily Phone Usage
- Social Media Usage
- Weekend Screen Time
- App Usage Count

---

## 📈 Output Metrics
The app produces:
### Pattern Deviation
Measures how different your lifestyle is from common patterns.
### Goal Alignment Score
Shows how close your lifestyle is to your selected goal.
### Smart Suggestions
Simulates habit changes and recommends the actions with the highest impact.

---

## 🖥️ Interface
The application is built using **Streamlit**, providing an interactive dashboard with sliders, goal selection, and visualizations.
---

## 🛠️ Tech Stack
Python  
PyTorch  
Streamlit  
Plotly  
NumPy  
Pandas  
Scikit-learn  
Joblib  
