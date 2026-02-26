import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Activity parameters
activities = {
    'resting': {
        'heart_rate': (60, 85),
        'temperature': (36.2, 36.8),
        'acc_x': (60, 130),
        'acc_y': (-60, 20),
        'acc_z': (960, 995),
        'count': 320
    },
    'walking': {
        'heart_rate': (80, 110),
        'temperature': (36.7, 37.3),
        'acc_x': (150, 400),
        'acc_y': (30, 220),
        'acc_z': (990, 1060),
        'count': 310
    },
    'running': {
        'heart_rate': (120, 150),
        'temperature': (37.0, 37.8),
        'acc_x': (500, 850),
        'acc_y': (150, 380),
        'acc_z': (1050, 1200),
        'count': 300
    }
}

# Generate data
data = []

for activity, params in activities.items():
    count = params['count']
    
    for _ in range(count):
        heart_rate = np.random.uniform(params['heart_rate'][0], params['heart_rate'][1])
        temperature = np.random.uniform(params['temperature'][0], params['temperature'][1])
        acc_x = np.random.uniform(params['acc_x'][0], params['acc_x'][1])
        acc_y = np.random.uniform(params['acc_y'][0], params['acc_y'][1])
        acc_z = np.random.uniform(params['acc_z'][0], params['acc_z'][1])
        
        # Add sensor noise (±2% typical microcontroller error)
        heart_rate += np.random.normal(0, 1.5)
        temperature += np.random.normal(0, 0.15)
        acc_x += np.random.normal(0, 5)
        acc_y += np.random.normal(0, 5)
        acc_z += np.random.normal(0, 5)
        
        data.append({
            'heart_rate': round(max(30, min(200, heart_rate)), 1),
            'temperature': round(max(35, min(40, temperature)), 1),
            'acc_x': round(max(0, min(1000, acc_x)), 1),
            'acc_y': round(max(-400, min(400, acc_y)), 1),
            'acc_z': round(max(800, min(1300, acc_z)), 1),
            'activity_label': activity
        })

# Add anomalies and false readings (sensor glitches, signal loss)
anomalies_count = 70

# Sensor spikes (sudden unrealistic values)
for _ in range(anomalies_count // 3):
    activity = np.random.choice(list(activities.keys()))
    data.append({
        'heart_rate': round(np.random.choice([np.random.uniform(200, 220), np.random.uniform(30, 50)]), 1),
        'temperature': round(np.random.choice([np.random.uniform(38, 40), np.random.uniform(35, 35.5)]), 1),
        'acc_x': round(np.random.uniform(0, 100), 1),
        'acc_y': round(np.random.uniform(-100, 100), 1),
        'acc_z': round(np.random.uniform(800, 900), 1),
        'activity_label': activity
    })

# Sensor dropout/zero values (connection loss)
for _ in range(anomalies_count // 3):
    activity = np.random.choice(list(activities.keys()))
    data.append({
        'heart_rate': round(np.random.uniform(50, 150), 1),
        'temperature': 0.0,  # Temperature sensor failure
        'acc_x': round(np.random.uniform(100, 500), 1),
        'acc_y': 0.0,  # Acc_y dropout
        'acc_z': round(np.random.uniform(900, 1100), 1),
        'activity_label': activity
    })

# Drift/calibration error (slightly offset readings)
for _ in range(anomalies_count // 3):
    activity = np.random.choice(list(activities.keys()))
    params = activities[activity]
    data.append({
        'heart_rate': round(np.random.uniform(params['heart_rate'][0] + 20, params['heart_rate'][1] + 20), 1),
        'temperature': round(np.random.uniform(params['temperature'][0] + 0.5, params['temperature'][1] + 0.5), 1),
        'acc_x': round(np.random.uniform(params['acc_x'][0] + 50, params['acc_x'][1] + 50), 1),
        'acc_y': round(np.random.uniform(params['acc_y'][0], params['acc_y'][1]), 1),
        'acc_z': round(np.random.uniform(params['acc_z'][0] + 30, params['acc_z'][1] + 30), 1),
        'activity_label': activity
    })

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV
df.to_csv('wearable_data.csv', index=False)

print("=" * 60)
print("✅ SENSOR DATA GENERATED FOR MICROCONTROLLER PROJECT")
print("=" * 60)
print(f"\n📊 Total Records: {len(df)}")
print(f"\n📈 Data Distribution:")
for activity in ['resting', 'walking', 'running']:
    count = (df['activity_label'] == activity).sum()
    pct = (count / len(df)) * 100
    print(f"   {activity.capitalize():12} {count:4} samples ({pct:5.1f}%)")

print(f"\n🔧 Data Features:")
print(f"   ✓ Realistic sensor noise (±2% error)")
print(f"   ✓ Sensor spikes/glitches ({anomalies_count // 3} readings)")
print(f"   ✓ Sensor dropout/zero values ({anomalies_count // 3} readings)")
print(f"   ✓ Calibration drift ({anomalies_count // 3} readings)")
print(f"\n📌 Suitable for microcontroller deployment with real-world robustness")
print(f"\n📋 Sample anomalies:")
print(df[df['temperature'] == 0.0].head(3))
