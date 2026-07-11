import os
import random
import kagglehub
import subprocess

print("Finding a female fingerprint...")
dataset_path = kagglehub.dataset_download("ruizgara/socofing")
real_images_dir = os.path.join(dataset_path, "SOCOFing", "Real")

female_files = []
for file in os.listdir(real_images_dir):
    if file.lower().endswith('.bmp'):
        parts = file.split('__')
        if len(parts) > 1:
            traits = parts[1].split('_')
            gender_str = traits[0]
            if gender_str == 'F':
                female_files.append(os.path.join(real_images_dir, file))

if not female_files:
    print("Could not find any female fingerprints in the dataset.")
else:
    # Pick a random female fingerprint
    test_image = random.choice(female_files)
    print(f"Selected: {test_image}")
    
    # Run the test script
    result = subprocess.run(['python', 'test_fingerprint.py', test_image], capture_output=True, text=True)
    print("\n--- TEST SCRIPT OUTPUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- ERRORS ---")
        print(result.stderr)
