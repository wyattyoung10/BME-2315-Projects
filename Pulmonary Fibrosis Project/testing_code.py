from pathlib import Path
import cv2, numpy as np, pandas as pd
from termcolor import colored
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

img_dir = Path(__file__).parent

# 1) Collect all candidate files
all_files = [str(p) for p in sorted(img_dir.glob("MASK_*.jpg"))]

# 2) EXACT NAMES of the two files to EXCLUDE from the main analysis (copy/paste from your printout!)
#    Note: We compare case-insensitively below, so caps won't matter.
check_files = {
    "MASK_SK658 Slobe ch010083.jpg",
    "MASK_SK658 Slobe ch010143.jpg",
}

# 3) Build base_filenames by excluding check_files (case-insensitive)
check_set_lower = {name.lower() for name in check_files}
base_filenames = [f for f in all_files if Path(f).name.lower() not in check_set_lower]

print("Base files:")
for f in base_filenames:
    print(f)
print("\n(Excluded from main analysis):")
for n in check_files:
    print(img_dir / n)
print()

# 4) Map depths by filename (so order can’t get scrambled)
#    >>>>>> MAKE SURE THESE KEYS MATCH EXACTLY WHAT YOU SEE PRINTED ABOVE <<<<<<
depth_map = {
    "MASK_SK658 Slobe ch010068.jpg": 9800,
    "MASK_SK658 Slobe ch010114.jpg": 9900,
    "MASK_SK658 Slobe ch010146.jpg": 2000,
    "MASK_Sk658 Llobe ch010024.jpg": 600,
    "MASK_Sk658 Llobe ch010039.jpg": 15,
    "MASK_Sk658 Llobe ch010071.jpg": 7100,
}

# Build the aligned depths list for the files we kept
try:
    depths = [depth_map[Path(f).name] for f in base_filenames]
except KeyError as e:
    missing = str(e).strip("'")
    raise SystemExit(
        f"\nDepth missing for filename: {missing}\n"
        f"Add it to depth_map with the correct value."
    )

# ==== Main analysis for base_filenames ====
images = []
white_counts, black_counts, white_percents = [], [], []

for filename in base_filenames:
    img = cv2.imread(filename, 0)
    images.append(img)

for x in range(len(base_filenames)):
    _, binary = cv2.threshold(images[x], 127, 255, cv2.THRESH_BINARY)
    white = np.sum(binary == 255)
    black = np.sum(binary == 0)
    white_counts.append(white)
    black_counts.append(black)

print(colored("Counts of pixel by color in each image", "yellow"))
for x in range(len(base_filenames)):
    print(colored(f"White pixels in image {x}: {white_counts[x]}", "white"))
    print(colored(f"Black pixels in image {x}: {black_counts[x]}", "black"))
    print()

for x in range(len(base_filenames)):
    white_percent = 100.0 * white_counts[x] / (white_counts[x] + black_counts[x])
    white_percents.append(white_percent)

print(colored("Percent white px:", "yellow"))
for x in range(len(base_filenames)):
    print(colored(f'{base_filenames[x]}:', "red"))
    print(f'{white_percents[x]:.6f}% White | Depth: {depths[x]} microns')
    print()

# Write main CSV
pd.DataFrame({
    'Filenames': [Path(f).name for f in base_filenames],
    'Depths': depths,
    'White percents': white_percents
}).to_csv('Percent_White_Pixels.csv', index=False)
print("CSV file 'Percent_White_Pixels.csv' has been created.")

# ==== Build interpolation function BEFORE checking extra depths ====
x = depths
y = white_percents
linetype = str(input(colored("Enter the function you wish to interpolate with: ", "yellow")))  # e.g., 'linear', 'quadratic', 'cubic'
i = interp1d(x, y, kind=linetype)

interpolate_depth = float(input(colored("Enter the depth at which you want to interpolate a point: ", "yellow")))
interpolate_point = float(i(interpolate_depth))
print(colored(f'The interpolated point is at x={interpolate_depth} and y={interpolate_point:.6f}.', "green"))

# ==== Check extra depths (2400, 2600) WITHOUT plotting or adding to main data ====
check_depths = [2400, 2600]
check_results = []
for d in check_depths:
    y_val = float(i(d))
    check_results.append((d, y_val))
    print(colored(f'Check depth {d}: interpolated value = {y_val:.6f}', "cyan"))

pd.DataFrame(check_results, columns=["Depths", "Interpolated White Percents"]).to_csv(
    'Interpolation_Check_Values.csv', index=False
)
print(colored("CSV file 'Interpolation_Check_Values.csv' created with check points.", "green"))

# ==== Plot (only base data + the one user-selected interpolated point) ====
depths_i = depths[:] + [interpolate_depth]
white_percents_i = white_percents[:] + [interpolate_point]

fig, axs = plt.subplots(2, 1)

axs[0].scatter(depths, white_percents, marker='o')
axs[0].set_title('Depth vs % white pixels')
axs[0].set_xlabel('depth (µm)')
axs[0].set_ylabel('% white pixels')
axs[0].set_ylim(0, 100)
axs[0].grid(True)

axs[1].scatter(depths_i, white_percents_i, marker='o')
axs[1].set_title('Depth vs % white pixels (with interpolated point)')
axs[1].set_xlabel('depth (µm)')
axs[1].set_ylabel('% white pixels')
axs[1].set_ylim(0, 100)
axs[1].grid(True)
axs[1].scatter(interpolate_depth, interpolate_point, s=100, label='Interpolated', zorder=3)

plt.tight_layout()
plt.show()
