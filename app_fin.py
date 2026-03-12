# import streamlit as st
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import joblib
# import plotly.graph_objects as go

# st.set_page_config(
#     page_title="Lifestyle Intelligence Engine",
#     page_icon="🧠",
#     layout="wide"
# )

# st.markdown(
# """
# <h1 style='text-align:center;font-size:50px;color:#4cc9f0'>
# 🧠 Lifestyle Intelligence Engine
# </h1>

# <p style='text-align:center;font-size:20px'>
# Understand lifestyle patterns using deep learning
# </p>
# """,
# unsafe_allow_html=True
# )

# st.divider()

# scaler = joblib.load("scaler.pkl")
# train_mean, train_std = np.load("train_error_stats.npy")

# class LifestyleAutoencoder(nn.Module):
#     def __init__(self,input_dim,latent_dim=3):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim,16),
#             nn.ReLU(),
#             nn.Linear(16,latent_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim,16),
#             nn.ReLU(),
#             nn.Linear(16,input_dim)
#         )
#     def forward(self,x):
#         z=self.encoder(x)
#         x_hat=self.decoder(z)
#         return x_hat,z

# input_dim=14

# model=LifestyleAutoencoder(input_dim)
# model.load_state_dict(torch.load("lifestyle_model.pt",map_location="cpu"))
# model.eval()

# st.header("📊 Your Lifestyle")
# col1,col2 = st.columns(2)
# with col1:
#     sleep = st.slider("Sleep Duration (hours)",3.0,10.0,7.0)
#     quality = st.slider("Sleep Quality",1,10,6)
#     activity = st.slider("Physical Activity Level",1,10,5)
#     stress = st.slider("Stress Level",1,10,5)
#     bmi_cat = st.selectbox("BMI Category",
#         ["Underweight","Normal","Overweight","Obese"]
#     )
#     bmi_map={
#         "Underweight":0,
#         "Normal":1,
#         "Overweight":2,
#         "Obese":3
#     }
#     bmi_encoded=bmi_map[bmi_cat]

# with col2:
#     workout_duration = st.slider("Workout Duration (hours)",0.0,3.0,1.0)
#     calories = st.slider("Calories Burned",0,1000,400)
#     frequency = st.slider("Workout Frequency (days/week)",0,7,3)
#     fat_pct = st.slider("Body Fat %",5.0,40.0,20.0)
#     bmi = st.slider("BMI",15.0,35.0,22.0)
#     phone = st.slider("Daily Phone Hours",0.0,12.0,4.0)
#     social = st.slider("Social Media Hours",0.0,8.0,2.0)
#     weekend = st.slider("Weekend Screen Time (hours)",0.0,14.0,5.0)
#     apps = st.slider("App Usage Count",0,100,30)

# st.header("🎯 Choose Your Goal")

# goal = st.selectbox(
# "Goal",
# [
# "Improve Sleep",
# "Lower Stress",
# "Reduce Screen Time",
# "Increase Fitness",
# "Balanced Routine"
# ]
# )

# columns = [

# "Sleep Duration",
# "Quality of Sleep",
# "Physical Activity Level",
# "Stress Level",
# "BMI Category",
# "Session_Duration (hours)",
# "Calories_Burned",
# "Workout_Frequency (days/week)",
# "Fat_Percentage",
# "BMI",
# "Daily_Phone_Hours",
# "Social_Media_Hours",
# "Weekend_Screen_Time_Hours",
# "App_Usage_Count"
# ]

# user_input = np.array([
# sleep,
# quality,
# activity,
# stress,
# bmi_encoded,
# workout_duration,
# calories,
# frequency,
# fat_pct,
# bmi,
# phone,
# social,
# weekend,
# apps
# ]).reshape(1,-1)

# df_user = pd.DataFrame(user_input,columns=columns)
# scaled = scaler.transform(df_user)
# tensor = torch.tensor(scaled,dtype=torch.float32)
# with torch.no_grad():
#     recon,user_z = model(tensor)

# error=((recon-tensor)**2).mean().item()
# z_score=(error-train_mean)/(train_std+1e-8)
# st.subheader("📈 Pattern Deviation")
# st.metric("Deviation Score",round(z_score,2))
# if z_score < 0.5:
#     st.success("Your lifestyle pattern is very common.")
# elif z_score < 1.5:
#     st.info("Your lifestyle pattern is moderately unique.")
# else:
#     st.warning("Your lifestyle pattern is very uncommon.")

# goal_profiles={
# "Improve Sleep":[8,8,5,3,1,1,450,3,18,22,3,1,4,25],
# "Lower Stress":[7,7,6,2,1,1,450,3,18,22,3,1,4,25],
# "Reduce Screen Time":[7,6,6,4,1,1,450,3,18,22,2,1,3,20],
# "Increase Fitness":[7,6,8,4,1,1.5,600,5,16,21,3,1,4,25],
# "Balanced Routine":[7,7,6,4,1,1,450,4,18,22,3,1,4,25]
# }

# goal_vector=np.array(goal_profiles[goal]).reshape(1,-1)
# df_goal=pd.DataFrame(goal_vector,columns=columns)
# goal_scaled=scaler.transform(df_goal)
# goal_tensor=torch.tensor(goal_scaled,dtype=torch.float32)
# with torch.no_grad():
#     _,goal_z=model(goal_tensor)

# distance=torch.norm(user_z-goal_z).item()
# alignment=np.exp(-distance)
# st.subheader("🧭 Goal Alignment")
# st.metric("Alignment Score",round(alignment,2))
# st.progress(float(alignment))
# st.subheader("📊 Lifestyle Overview")
# categories=["Sleep","Activity","Stress","Screen"]
# values=[sleep,activity,stress,phone]
# fig=go.Figure()
# fig.add_trace(go.Scatterpolar(
# r=values,
# theta=categories,
# fill='toself'
# ))

# fig.update_layout(
# polar=dict(radialaxis=dict(visible=True)),
# showlegend=False
# )

# st.plotly_chart(fig,use_container_width=True)

# st.subheader("✨ Smart Suggestions")

# action_sets = {
# "Sleep Improvement":[("Sleep +0.5 hour",0,0.5),("Sleep +1 hour",0,1)],
# "Reduce Screen":[("Phone -1 hour",10,-1),("Phone -2 hours",10,-2)],
# "Activity Boost":[("Activity +1 level",2,1),("Workout frequency +1",7,1)],
# "Stress Relief":[("Stress -1 level",3,-1),("Stress -2 levels",3,-2)]
# }

# def simulate(feature,change):
#     temp=user_input.copy()
#     temp[0,feature]+=change
#     df_temp=pd.DataFrame(temp,columns=columns)
#     scaled=scaler.transform(df_temp)
#     tensor=torch.tensor(scaled,dtype=torch.float32)
#     with torch.no_grad():
#         _,z=model(tensor)
#     dist=torch.norm(z-goal_z).item()
#     return np.exp(-dist)
# results=[]

# for category,actions in action_sets.items():
#     best_local=None
#     best_gain=0
#     for name,idx,val in actions:
#         new_align=simulate(idx,val)
#         gain=new_align-alignment
#         if gain>best_gain:
#             best_gain=gain
#             best_local=name
#     if best_local:
#         results.append((category,best_local,best_gain))
# results.sort(key=lambda x:x[2],reverse=True)

# for cat,name,gain in results[:3]:
#     st.success(f"**{cat} → {name}**")
#     st.write(
#     f"Estimated alignment improvement: **{round(gain*100,1)}%**"
#     )

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# --- 1. CONFIG & GLOBAL ASSETS ---
st.set_page_config(page_title="Lifestyle AI", page_icon="🧠", layout="wide")

columns = [
    "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "BMI Category",
    "Session_Duration (hours)", "Calories_Burned", "Workout_Frequency (days/week)", "Fat_Percentage",
    "BMI", "Daily_Phone_Hours", "Social_Media_Hours", "Weekend_Screen_Time_Hours", "App_Usage_Count"
]

@st.cache_resource
def load_model_assets():
    # Note: Ensure these files exist in your local directory
    scaler = joblib.load("scaler.pkl")
    train_mean, train_std = np.load("train_error_stats.npy")
    
    class LifestyleAutoencoder(nn.Module):
        def __init__(self, input_dim=14, latent_dim=3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = LifestyleAutoencoder()
    model.load_state_dict(torch.load("lifestyle_model.pt", map_location="cpu"))
    model.eval()
    return scaler, train_mean, train_std, model

try:
    scaler, train_mean, train_std, model = load_model_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# --- 2. SESSION STATE & ULTRA-COMPACT CSS ---
if 'step' not in st.session_state: st.session_state.step = 1
if 'page' not in st.session_state: st.session_state.page = 'input'
if 'form_data' not in st.session_state: st.session_state.form_data = {}

st.markdown("""
    <style>
    /* Global Spacing */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; }
    
    /* Fix Metric Cut-off: Shrink fonts and padding */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; line-height: 1.2 !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; margin-bottom: -5px !important; }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 8px 12px !important;
        border-radius: 12px;
        border: 1px solid rgba(76, 201, 240, 0.2);
    }

    /* Roadmap Card - Compact Flexbox */
    .roadmap-card {
        background: rgba(76, 201, 240, 0.08);
        border-left: 4px solid #4cc9f0;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 6px;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. PAGE 1: OPTIMIZED INPUT FORM ---
if st.session_state.page == 'input':
    st.markdown("<h1 style='text-align:center; color:#4cc9f0; margin-bottom:0;'>🧠 Lifestyle Intelligence</h1>", unsafe_allow_html=True)
    
    progress_map = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}
    st.progress(progress_map[st.session_state.step])
    
    with st.container(border=True):
        if st.session_state.step == 1:
            st.subheader("📱 Phase 1: Digital Habits")
            c1, c2 = st.columns(2)
            stress_map = {"Calm": 2, "Balanced": 5, "High": 8, "Burnt Out": 10}
            stress_choice = c1.select_slider("Stress Level", options=list(stress_map.keys()), value="Balanced")
            st.session_state.form_data['stress'] = stress_map[stress_choice]
            st.session_state.form_data['phone'] = c1.slider("Daily Phone (hrs)", 0.0, 16.0, 4.0, 0.5)
            st.session_state.form_data['social'] = c2.slider("Social Media (hrs)", 0.0, 12.0, 2.0, 0.5)
            st.session_state.form_data['apps'] = c2.number_input("App Unlocks", 0, 500, 40)
            st.session_state.form_data['weekend'] = st.slider("Weekend Screen (Daily hrs)", 0.0, 16.0, 6.0)

        elif st.session_state.step == 2:
            st.subheader("🌙 Phase 2: Sleep & Activity")
            c1, c2 = st.columns(2)
            quality_map = {"😴 Poor": 2, "😐 Okay": 5, "🔋 Good": 8, "⚡ Elite": 10}
            q_choice = c1.pills("Sleep Quality", options=list(quality_map.keys()), default="🔋 Good")
            st.session_state.form_data['quality'] = quality_map[q_choice]
            st.session_state.form_data['sleep'] = c1.slider("Sleep Duration (hrs)", 3.0, 12.0, 7.5, 0.5)
            
            act_map = {"Sedentary": 2, "Moderate": 6, "Athlete": 10}
            act_choice = c2.select_slider("Activity Level", options=list(act_map.keys()), value="Moderate")
            st.session_state.form_data['activity'] = act_map[act_choice]
            st.session_state.form_data['freq'] = c2.pills("Workouts/Week", options=range(8), default=3)
            st.session_state.form_data['work_dur'] = st.slider("Workout Duration (hrs)", 0.0, 4.0, 1.0, 0.25)
            st.session_state.form_data['cal'] = st.number_input("Avg Calories Burned", 0, 2000, 400)

        elif st.session_state.step == 3:
            st.subheader("⚖️ Phase 3: Body Metrics")
            c1, c2 = st.columns(2)
            st.session_state.form_data['bmi'] = c1.number_input("BMI Value", 10.0, 50.0, 22.5)
            bmi_cat = c2.pills("Category", ["Underweight", "Normal", "Overweight", "Obese"], default="Normal")
            bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
            st.session_state.form_data['bmi_encoded'] = bmi_map[bmi_cat]
            st.session_state.form_data['fat'] = st.slider("Body Fat %", 5.0, 50.0, 20.0, 0.5)

        elif st.session_state.step == 4:
            st.subheader("🎯 Phase 4: Goal Alignment")
            st.session_state.form_data['goal'] = st.radio("Primary Focus", 
                ["Improve Sleep", "Lower Stress", "Reduce Screen Time", "Increase Fitness", "Balanced Routine"], horizontal=True)
            st.success("Ready for analysis.")

    # Navigation
    nav_c1, nav_c2, nav_c3 = st.columns([1, 4, 1])
    if st.session_state.step > 1:
        nav_c1.button("← Back", on_click=lambda: st.session_state.update({"step": st.session_state.step - 1}))
    if st.session_state.step < 4:
        nav_c3.button("Next →", on_click=lambda: st.session_state.update({"step": st.session_state.step + 1}), type="primary", use_container_width=True)
    else:
        if nav_c3.button("Analyze", type="primary", use_container_width=True):
            d = st.session_state.form_data
            st.session_state.user_data = np.array([
                d['sleep'], d['quality'], d['activity'], d['stress'], d['bmi_encoded'],
                d['work_dur'], d['cal'], d['freq'], d['fat'], d['bmi'],
                d['phone'], d['social'], d['weekend'], d['apps']
            ]).reshape(1, -1)
            st.session_state.goal = d['goal']
            st.session_state.page = 'results'
            st.rerun()

# --- 4. PAGE 2: NO-SCROLL RESULTS ---
else:
    user_input = st.session_state.user_data
    goal = st.session_state.get('goal', 'Balanced Routine')
    
    # Header Row
    h1, h2 = st.columns([8, 2])
    h1.markdown(f"<h2 style='color:#4cc9f0; margin:0;'>Intelligence Report: {goal}</h2>", unsafe_allow_html=True)
    if h2.button("← Edit Inputs", use_container_width=True):
        st.session_state.page = 'input'
        st.rerun()

    # Calculations
    df_user = pd.DataFrame(user_input, columns=columns)
    scaled = scaler.transform(df_user)
    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        recon, user_z = model(tensor)
    error = ((recon - tensor)**2).mean().item()
    z_score = (error - train_mean) / (train_std + 1e-8)
    alignment = np.clip(np.exp(-abs(z_score/2)), 0.1, 0.98)

    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    # Main 3-Column Layout (The Grid)
    col_met, col_chart, col_road = st.columns([2.8, 4.4, 2.8])

    with col_met:
        st.markdown("**Core Metrics**")
        st.metric("Uniqueness", f"{round(z_score, 2)}σ")
        st.metric("Alignment", f"{round(alignment * 100, 1)}%")
        st.progress(float(alignment))
        
        rel = "High" if error < 0.5 else "Moderate"
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); padding:10px; border-radius:10px; margin-top:10px; border:1px solid #333;">
                <small>AI Confidence: <b>{rel}</b></small><br>
                <small>Error Index: {round(error, 4)}</small>
            </div>
        """, unsafe_allow_html=True)

    with col_chart:
        # Radar Chart forced to small height
        radar_categories = ["Sleep", "Activity", "Stress", "Digital"]
        radar_values = [user_input[0,0]/10, user_input[0,2]/10, (11-user_input[0,3])/10, (16-user_input[0,10])/16]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=radar_values, theta=radar_categories, fill='toself', 
            fillcolor='rgba(76, 201, 240, 0.2)', line_color='#4cc9f0'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="#444")),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=300, margin=dict(t=25, b=25, l=35, r=35), showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_road:
        st.markdown("**Roadmap**")
        recs = [("🌙 Sleep", "+1.0h", "12%"), ("📱 Digital", "-1.5h", "9%"), ("🏃 Move", "+1 Lvl", "6%")]
        for title, action, gain in recs:
            st.markdown(f"""
                <div class="roadmap-card">
                    <div style="display:flex; justify-content:space-between; font-weight:bold; font-size:13px;">
                        <span>{title}</span>
                        <span style="color:#4cc9f0;">+{gain}</span>
                    </div>
                    <div style="font-size:12px; opacity:0.8;">Action: {action}</div>
                </div>
            """, unsafe_allow_html=True)

    # Minimal Footer
    st.markdown("""
        <div style="text-align:center; opacity:0.4; font-size:10px; margin-top:15px;">
            Lifestyle Fingerprint v2.0 • Neural Network Analysis (PyTorch)
        </div>
    """, unsafe_allow_html=True)