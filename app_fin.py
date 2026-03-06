import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Lifestyle Intelligence Engine",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
"""
<h1 style='text-align:center;font-size:50px;color:#4cc9f0'>
🧠 Lifestyle Intelligence Engine
</h1>

<p style='text-align:center;font-size:20px'>
Understand lifestyle patterns using deep learning
</p>
""",
unsafe_allow_html=True
)

st.divider()

scaler = joblib.load("scaler.pkl")
train_mean, train_std = np.load("train_error_stats.npy")

class LifestyleAutoencoder(nn.Module):
    def __init__(self,input_dim,latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,16),
            nn.ReLU(),
            nn.Linear(16,latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,16),
            nn.ReLU(),
            nn.Linear(16,input_dim)
        )
    def forward(self,x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat,z

input_dim=14

model=LifestyleAutoencoder(input_dim)
model.load_state_dict(torch.load("lifestyle_model.pt",map_location="cpu"))
model.eval()

st.header("📊 Your Lifestyle")
col1,col2 = st.columns(2)
with col1:
    sleep = st.slider("Sleep Duration (hours)",3.0,10.0,7.0)
    quality = st.slider("Sleep Quality",1,10,6)
    activity = st.slider("Physical Activity Level",1,10,5)
    stress = st.slider("Stress Level",1,10,5)
    bmi_cat = st.selectbox("BMI Category",
        ["Underweight","Normal","Overweight","Obese"]
    )
    bmi_map={
        "Underweight":0,
        "Normal":1,
        "Overweight":2,
        "Obese":3
    }
    bmi_encoded=bmi_map[bmi_cat]

with col2:
    workout_duration = st.slider("Workout Duration (hours)",0.0,3.0,1.0)
    calories = st.slider("Calories Burned",0,1000,400)
    frequency = st.slider("Workout Frequency (days/week)",0,7,3)
    fat_pct = st.slider("Body Fat %",5.0,40.0,20.0)
    bmi = st.slider("BMI",15.0,35.0,22.0)
    phone = st.slider("Daily Phone Hours",0.0,12.0,4.0)
    social = st.slider("Social Media Hours",0.0,8.0,2.0)
    weekend = st.slider("Weekend Screen Time (hours)",0.0,14.0,5.0)
    apps = st.slider("App Usage Count",0,100,30)

st.header("🎯 Choose Your Goal")

goal = st.selectbox(
"Goal",
[
"Improve Sleep",
"Lower Stress",
"Reduce Screen Time",
"Increase Fitness",
"Balanced Routine"
]
)

columns = [

"Sleep Duration",
"Quality of Sleep",
"Physical Activity Level",
"Stress Level",
"BMI Category",
"Session_Duration (hours)",
"Calories_Burned",
"Workout_Frequency (days/week)",
"Fat_Percentage",
"BMI",
"Daily_Phone_Hours",
"Social_Media_Hours",
"Weekend_Screen_Time_Hours",
"App_Usage_Count"
]

user_input = np.array([
sleep,
quality,
activity,
stress,
bmi_encoded,
workout_duration,
calories,
frequency,
fat_pct,
bmi,
phone,
social,
weekend,
apps
]).reshape(1,-1)

df_user = pd.DataFrame(user_input,columns=columns)
scaled = scaler.transform(df_user)
tensor = torch.tensor(scaled,dtype=torch.float32)
with torch.no_grad():
    recon,user_z = model(tensor)

error=((recon-tensor)**2).mean().item()
z_score=(error-train_mean)/(train_std+1e-8)
st.subheader("📈 Pattern Deviation")
st.metric("Deviation Score",round(z_score,2))
if z_score < 0.5:
    st.success("Your lifestyle pattern is very common.")
elif z_score < 1.5:
    st.info("Your lifestyle pattern is moderately unique.")
else:
    st.warning("Your lifestyle pattern is very uncommon.")

goal_profiles={
"Improve Sleep":[8,8,5,3,1,1,450,3,18,22,3,1,4,25],
"Lower Stress":[7,7,6,2,1,1,450,3,18,22,3,1,4,25],
"Reduce Screen Time":[7,6,6,4,1,1,450,3,18,22,2,1,3,20],
"Increase Fitness":[7,6,8,4,1,1.5,600,5,16,21,3,1,4,25],
"Balanced Routine":[7,7,6,4,1,1,450,4,18,22,3,1,4,25]
}

goal_vector=np.array(goal_profiles[goal]).reshape(1,-1)
df_goal=pd.DataFrame(goal_vector,columns=columns)
goal_scaled=scaler.transform(df_goal)
goal_tensor=torch.tensor(goal_scaled,dtype=torch.float32)
with torch.no_grad():
    _,goal_z=model(goal_tensor)

distance=torch.norm(user_z-goal_z).item()
alignment=np.exp(-distance)
st.subheader("🧭 Goal Alignment")
st.metric("Alignment Score",round(alignment,2))
st.progress(float(alignment))
st.subheader("📊 Lifestyle Overview")
categories=["Sleep","Activity","Stress","Screen"]
values=[sleep,activity,stress,phone]
fig=go.Figure()
fig.add_trace(go.Scatterpolar(
r=values,
theta=categories,
fill='toself'
))

fig.update_layout(
polar=dict(radialaxis=dict(visible=True)),
showlegend=False
)

st.plotly_chart(fig,use_container_width=True)

st.subheader("✨ Smart Suggestions")

action_sets = {
"Sleep Improvement":[("Sleep +0.5 hour",0,0.5),("Sleep +1 hour",0,1)],
"Reduce Screen":[("Phone -1 hour",10,-1),("Phone -2 hours",10,-2)],
"Activity Boost":[("Activity +1 level",2,1),("Workout frequency +1",7,1)],
"Stress Relief":[("Stress -1 level",3,-1),("Stress -2 levels",3,-2)]
}

def simulate(feature,change):
    temp=user_input.copy()
    temp[0,feature]+=change
    df_temp=pd.DataFrame(temp,columns=columns)
    scaled=scaler.transform(df_temp)
    tensor=torch.tensor(scaled,dtype=torch.float32)
    with torch.no_grad():
        _,z=model(tensor)
    dist=torch.norm(z-goal_z).item()
    return np.exp(-dist)
results=[]

for category,actions in action_sets.items():
    best_local=None
    best_gain=0
    for name,idx,val in actions:
        new_align=simulate(idx,val)
        gain=new_align-alignment
        if gain>best_gain:
            best_gain=gain
            best_local=name
    if best_local:
        results.append((category,best_local,best_gain))
results.sort(key=lambda x:x[2],reverse=True)

for cat,name,gain in results[:3]:
    st.success(f"**{cat} → {name}**")
    st.write(
    f"Estimated alignment improvement: **{round(gain*100,1)}%**"
    )