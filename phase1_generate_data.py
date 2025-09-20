import pandas as pd
import numpy as np
import math
from tqdm import tqdm # A library to show a progress bar

# --- Engineering Constants and Material Properties ---
# We'll use properties for hardened C45 steel as in our example
ALLOWABLE_BENDING_STRESS = 200  # MPa (N/mm^2)
SURFACE_HARDNESS = 300          # BHN
PRESSURE_ANGLE = 20             # degrees
PINION_TEETH = 18               # Minimum teeth to avoid interference

# Standard gear modules (tooth sizes) in mm
STANDARD_MODULES = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20]

def design_one_gear(power_kw, speed_rpm, ratio):
    """
    Designs a single spur gear pair for given operating conditions.
    Returns [module, pinion_teeth, face_width] if successful, otherwise None.
    """
    if speed_rpm == 0: return None
    
    # --- Step 1: Initial Calculations ---
    torque_nm = (power_kw * 9550) / speed_rpm
    torque_n_mm = torque_nm * 1000
    
    gear_teeth = PINION_TEETH * ratio
    
    # Lewis Form Factor (Y) for the pinion
    lewis_form_factor = 0.154 - (0.912 / PINION_TEETH)
    
    # --- Step 2: Iterative Design ---
    # Loop through standard modules to find the first one that is safe
    for module in STANDARD_MODULES:
        # Assume face width is 10 times the module
        face_width = 10 * module
        
        # --- Geometry and Velocity ---
        pitch_diameter = module * PINION_TEETH
        if pitch_diameter == 0: continue
        
        pitch_velocity_ms = (math.pi * pitch_diameter * speed_rpm) / (60 * 1000)
        
        # --- Load Calculations ---
        tangential_load_Ft = (2 * torque_n_mm) / pitch_diameter
        
        # Velocity Factor (Cv) for dynamic load
        # Using a common formula for cut teeth
        if pitch_velocity_ms < 10:
            Cv = 3.05 / (3.05 + pitch_velocity_ms)
        else:
            Cv = 6.1 / (6.1 + pitch_velocity_ms)
        
        effective_load_Feff = tangential_load_Ft / Cv
        
        # --- Strength Checks ---
        # 1. Bending Strength (Lewis Equation)
        beam_strength_Fb = (ALLOWABLE_BENDING_STRESS * face_width * math.pi * module * lewis_form_factor)

        # 2. Wear Strength (Buckingham Equation)
        ratio_factor_Q = (2 * gear_teeth) / (PINION_TEETH + gear_teeth)
        load_stress_factor_K = 0.16 * (SURFACE_HARDNESS / 100)**2
        wear_strength_Fw = pitch_diameter * face_width * ratio_factor_Q * load_stress_factor_K

        # --- Verdict ---
        # Check if the gear is strong enough for the dynamic load
        if beam_strength_Fb > effective_load_Feff and wear_strength_Fw > effective_load_Feff:
            # SUCCESS: This module is safe.
            return [module, PINION_TEETH, face_width]

    # FAIL: No suitable module found in the standard list for these extreme conditions
    return None


    # --- Main Data Generation Script ---
NUM_SAMPLES = 100000  # Let's create 100000 successful designs
expert_data = []

print(f"Generating {NUM_SAMPLES} gear designs. This may take a few minutes...")

# Use tqdm for a nice progress bar
with tqdm(total=NUM_SAMPLES) as pbar:
    while len(expert_data) < NUM_SAMPLES:
        # Generate random but realistic operating conditions
        p_in = np.random.uniform(1, 150)    # 1 to 150 kW
        n_in = np.random.uniform(500, 3000) # 500 to 3000 RPM
        i = np.random.uniform(1, 6)         # Ratio from 1 to 6
        
        # Call our expert designer
        result = design_one_gear(power_kw=p_in, speed_rpm=n_in, ratio=i)
        
        # If the design was successful, save it
        if result is not None:
            expert_data.append([p_in, n_in, i] + result)
            pbar.update(1) # Update the progress bar

# Create the final DataFrame
columns = ['Input_Power_kW', 'Input_Speed_RPM', 'Velocity_Ratio', 
           'Result_Module', 'Result_Teeth', 'Result_FaceWidth']
df = pd.DataFrame(expert_data, columns=columns)

# Save to CSV
df.to_csv('expert_gear_solutions.csv', index=False)

print("\n----------------------------------------------------")
print("✅ Phase 1 Complete!")
print(f"Dataset 'expert_gear_solutions.csv' with {len(df)} samples has been created.")
print("A quick preview of the data:")
print(df.head())