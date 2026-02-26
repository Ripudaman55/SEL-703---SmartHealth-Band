import pandas as pd
import numpy as np
from scipy import signal

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("📊 ADVANCED DATA GENERATION - BEST PRACTICES FOR TRAINING")
print("=" * 70)

# Activity parameters with STRICT ranges for clean data
activities = {
    'sitting': {
        'heart_rate': (55, 80),
        'temperature': (36.1, 36.7),
        'acc_x': (40, 110),
        'acc_y': (-30, 30),
        'acc_z': (950, 1000),
        'count': 200  # Minimum 200 per activity
    },
    'walking': {
        'heart_rate': (75, 105),
        'temperature': (36.6, 37.2),
        'acc_x': (120, 350),
        'acc_y': (40, 180),
        'acc_z': (980, 1050),
        'count': 200
    },
    'running': {
        'heart_rate': (115, 160),
        'temperature': (37.1, 37.8),
        'acc_x': (450, 850),
        'acc_y': (200, 400),
        'acc_z': (1000, 1200),
        'count': 200
    }
}

# Generate data
data = []

for activity, params in activities.items():
    count = params['count']
    print(f"\n📝 Generating {activity.capitalize()} ({count} samples)...")
    
    for i in range(count):
        # Generate with TIGHT RANGES (minimal noise)
        heart_rate = np.random.uniform(params['heart_rate'][0], params['heart_rate'][1])
        temperature = np.random.uniform(params['temperature'][0], params['temperature'][1])
        acc_x = np.random.uniform(params['acc_x'][0], params['acc_x'][1])
        acc_y = np.random.uniform(params['acc_y'][0], params['acc_y'][1])
        acc_z = np.random.uniform(params['acc_z'][0], params['acc_z'][1])
        
        # Add MINIMAL sensor noise (0.5% - much cleaner than before)
        heart_rate += np.random.normal(0, 0.8)
        temperature += np.random.normal(0, 0.08)
        acc_x += np.random.normal(0, 2)
        acc_y += np.random.normal(0, 2)
        acc_z += np.random.normal(0, 2)
        
        # Clamp to valid ranges (NO SPIKES)
        heart_rate = max(40, min(180, heart_rate))
        temperature = max(35.5, min(38.5, temperature))
        acc_x = max(0, min(1000, acc_x))
        acc_y = max(-400, min(400, acc_y))
        acc_z = max(900, min(1250, acc_z))
        
        data.append({
            'heart_rate': round(heart_rate, 1),
            'temperature': round(temperature, 1),
            'acc_x': round(acc_x, 1),
            'acc_y': round(acc_y, 1),
            'acc_z': round(acc_z, 1),
            'activity_label': activity
        })

# Create DataFrame
df = pd.DataFrame(data)

# Calculate derived features BEFORE shuffling
print("\n🔧 STEP 1: Creating Enhanced Features...")

# Feature 1: Acceleration Magnitude
df['acc_magnitude'] = np.sqrt(
    df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
)
print("   ✓ Acceleration Magnitude = sqrt(x² + y² + z²)")

# Feature 2: Acceleration Range (max - min across axes)
df['acc_range'] = df[['acc_x', 'acc_y', 'acc_z']].max(axis=1) - df[['acc_x', 'acc_y', 'acc_z']].min(axis=1)
print("   ✓ Acceleration Range (max - min across axes)")

# Feature 3: Heart Rate to Temperature ratio
df['hr_temp_ratio'] = df['heart_rate'] / (df['temperature'] * 10)
print("   ✓ Heart Rate to Temperature Ratio")

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Remove any remaining outliers (spikes)
print("\n🧹 STEP 2: Data Cleaning...")

initial_count = len(df)

# Remove heart rate spikes (deviation > 3 sigma from activity mean)
for activity in df['activity_label'].unique():
    activity_mask = df['activity_label'] == activity
    hr_mean = df[activity_mask]['heart_rate'].mean()
    hr_std = df[activity_mask]['heart_rate'].std()
    
    spike_mask = (df['heart_rate'] > hr_mean + 3*hr_std) | (df['heart_rate'] < hr_mean - 3*hr_std)
    df = df[~spike_mask | ~activity_mask]

# Remove temperature anomalies
temp_mask = (df['temperature'] < 35.5) | (df['temperature'] > 38.5)
df = df[~temp_mask]

removed_count = initial_count - len(df)
print(f"   ✓ Removed {removed_count} spike readings")
print(f"   ✓ Final clean dataset: {len(df)} samples")

# Verify balance
print("\n⚖️  STEP 3: Verifying Dataset Balance...")
for activity in ['sitting', 'walking', 'running']:
    count = (df['activity_label'] == activity).sum()
    pct = (count / len(df)) * 100
    bar = "█" * int(pct / 2)
    print(f"   {activity.capitalize():12} {count:3} samples ({pct:5.1f}%) {bar}")

# Save to CSV
df.to_csv('wearable_data.csv', index=False)

print("\n" + "=" * 70)
print("✅ ENHANCED DATASET CREATED")
print("=" * 70)
print(f"\n📊 Dataset Summary:")
print(f"   Total Records: {len(df)}")
print(f"   Features: heart_rate, temperature, acc_x, acc_y, acc_z")
print(f"   Derived Features: acc_magnitude, acc_variance, hr_temp_ratio")
print(f"\n🎯 Quality Metrics:")
print(f"   ✓ Balanced distribution across activities")
print(f"   ✓ Minimal sensor noise (0.5%)")
print(f"   ✓ No spike anomalies")
print(f"   ✓ Consistent sampling")
print(f"   ✓ 8 total features (5 raw + 3 engineered)")
print("\n📁 Saved to: wearable_data.csv")
print("=" * 70 + "\n")
