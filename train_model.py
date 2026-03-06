import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

sleep_df = pd.read_csv(r"DataSets\Sleep_health_and_lifestyle_dataset.csv")
fitness_df = pd.read_csv(r"DataSets\gym_members_exercise_tracking_synthetic_data.csv")
screen_df = pd.read_csv(r"DataSets\Smartphone_Usage_Productivity_Dataset_50000.csv")

sleep_df.columns = sleep_df.columns.str.strip()
fitness_df.columns = fitness_df.columns.str.strip()
screen_df.columns = screen_df.columns.str.strip()

sleep_features = sleep_df[
    [
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "BMI Category",
    ]
].copy()

sleep_features["BMI Category"] = sleep_features["BMI Category"].map(
    {
        "Underweight": 0,
        "Normal": 1,
        "Overweight": 2,
        "Obese": 3,
    }
).fillna(1)

n_samples = len(sleep_features)

fitness_sample = fitness_df.sample(
    n=n_samples, replace=True, random_state=42
).reset_index(drop=True)

fitness_features = fitness_sample[
    [
        "Session_Duration (hours)",
        "Calories_Burned",
        "Workout_Frequency (days/week)",
        "Fat_Percentage",
        "BMI",
    ]
]

screen_sample = screen_df.sample(
    n=n_samples, replace=True, random_state=42
).reset_index(drop=True)

screen_features = screen_sample[
    [
        "Daily_Phone_Hours",
        "Social_Media_Hours",
        "Weekend_Screen_Time_Hours",
        "App_Usage_Count",
    ]
]

X = pd.concat(
    [
        sleep_features.reset_index(drop=True),
        fitness_features.reset_index(drop=True),
        screen_features.reset_index(drop=True),
    ],
    axis=1,
)
X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median())

assert not X.isnull().any().any(), "❌ NaNs still present in X!"

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
assert np.isfinite(X_scaled).all(), "❌ Non-finite values after scaling!"
joblib.dump(scaler, "scaler.pkl")
class LifestyleAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
model = LifestyleAutoencoder(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 50
noise_std = 0.05

for epoch in range(epochs):
    total_loss = 0.0
    for (x,) in loader:
        noise = noise_std * torch.randn_like(x)
        x_noisy = x + noise
        x_hat, _ = model(x_noisy)
        loss = criterion(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "lifestyle_model.pt")

with torch.no_grad():
    recon, _ = model(X_tensor)
    train_error = ((recon - X_tensor) ** 2).mean(dim=1).numpy()

mean_err = train_error.mean()
std_err = train_error.std()

print("\nTRAINING ERROR STATS")
print("--------------------")
print(f"Mean error: {mean_err:.4f}")
print(f"Std  error: {std_err:.4f}")

np.save("train_error_stats.npy", np.array([mean_err, std_err]))
print("\n✅ Training complete. Model is healthy and usable.")