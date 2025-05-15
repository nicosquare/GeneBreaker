import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
# Patch is no longer needed as legend is removed
# from matplotlib.patches import Patch
from Bio import SeqIO
import collections
import random # Import the random module for sampling

# --- Configuration ---
# Define the base directory where the virus datasets are stored
# IMPORTANT: Update this path to the actual location of your virus data folders.
base_dir = "./"

# Map each virus directory name to its broader category (Should match pie chart script)
virus_category_map = {
    "Adenoviridae": "Large DNA viruses",
    "VARV": "Large DNA viruses",
    "Herpesviridae": "Large DNA viruses",
    "HPV": "Large DNA viruses",
    "B19": "Small DNA viruses",
    "Reovirus": "Double-stranded RNA viruses",
    "SARS-CoV-2": "Positive-strand RNA viruses",
    "MERS-CoV": "Positive-strand RNA viruses",
    "Coronavirus229E": "Positive-strand RNA viruses",
    "CoronavirusHKU1": "Positive-strand RNA viruses",
    "CoronavirusNL63": "Positive-strand RNA viruses",
    "CoronavirusOC43": "Positive-strand RNA viruses",
    "Denguevirus": "Positive-strand RNA viruses",
    "HCV": "Positive-strand RNA viruses",
    "Japanese encephalitis virus": "Positive-strand RNA viruses",
    "HIV": "Positive-strand RNA viruses",
    "Influenza": "Negative-strand RNA viruses",
    "Measles virus": "Negative-strand RNA viruses",
    "Mumpsvirus": "Negative-strand RNA viruses",
    "Rabiesvirus": "Negative-strand RNA viruses",
    "Norovirus": "Enteric RNA viruses",
    "Poliovirus": "Enteric RNA viruses",
    "Varicella-Zoster Virus": "Large DNA viruses"
    # Add more mappings if needed
}

# --- Constants ---
# CDS_COUNT_CAP is still needed to maintain the category sort order from the pie chart.
CDS_COUNT_CAP = 3

# --- Data Processing ---

# Store details: {'name': virus_name, 'category': category,
#                 'random_length': sampled_length, 'all_lengths': cds_lengths,
#                 'cds_count': original_cds_count}
virus_data = []
# We still need category counts (capped) to sort categories consistently with the pie chart
category_cds_counts_capped = collections.defaultdict(int)

print(f"Starting analysis for randomly sampled CDS length in base directory: {os.path.abspath(base_dir)}")

# Process viruses to get lengths and counts
for item_name in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item_name)
    if os.path.isdir(item_path) and item_name in virus_category_map:
        virus_name = item_name
        category = virus_category_map[virus_name]
        patho_path = os.path.join(item_path, "patho")
        gb_file_dir = patho_path if os.path.isdir(patho_path) else item_path

        try:
            files = [os.path.join(gb_file_dir, f) for f in os.listdir(gb_file_dir) if f.endswith((".gb", ".gbk", ".genbank"))]
        except FileNotFoundError: continue
        except Exception as e:
             print(f"Warning: Error listing files in {gb_file_dir}: {e}")
             continue
        if not files: continue

        print(f"Processing {len(files)} file(s) for virus: {virus_name}...")

        cds_lengths = [] # Store all CDS lengths for this virus
        original_cds_count = 0

        for file_path in files:
            try:
                for record in SeqIO.parse(file_path, "genbank"):
                    for feature in record.features:
                        if feature.type == "CDS":
                            try:
                                length = len(feature.location)
                                cds_lengths.append(length)
                                original_cds_count += 1
                            except TypeError:
                                print(f"Warning: Could not determine length for a CDS feature in {file_path}. Skipping feature.")
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")

        # Randomly sample one length if available
        sampled_length = random.choice(cds_lengths) if cds_lengths else 0
        # Calculate capped count for category sorting consistency
        capped_cds_count = min(original_cds_count, CDS_COUNT_CAP)

        if original_cds_count > 0: # Only include viruses where CDS were found
            virus_data.append({
                'name': virus_name,
                'category': category,
                'random_length': sampled_length, # Store the randomly sampled length
                'all_lengths': cds_lengths, # Store all lengths for reference if needed
                'cds_count': original_cds_count,
            })
            # Update category capped counts for sorting categories later
            if capped_cds_count > 0:
                 category_cds_counts_capped[category] += capped_cds_count
        else:
            print(f"Info: No CDS regions found for virus '{virus_name}'. Excluding from bar plot.")


# --- Data Preparation for Plotting ---

if not virus_data:
    print("Error: No virus data with CDS found. Cannot generate bar plot.")
    exit()

# Sort categories by total capped CDS count (descending) - same order as pie chart
sorted_categories = collections.OrderedDict(
    sorted(category_cds_counts_capped.items(), key=lambda item: item[1], reverse=True)
)
category_order_map = {category: i for i, category in enumerate(sorted_categories.keys())}

# Define base colors for categories (consistent with pie chart)
num_categories = len(sorted_categories)
colors = plt.cm.tab20(np.linspace(0, 1, min(num_categories, 20)))
if num_categories > 20:
     colors = plt.cm.tab20(np.linspace(0, 1, 20))
     inner_colors = [colors[i % 20] for i in range(num_categories)]
     print("Warning: More than 20 categories, colors will repeat.")
else:
     inner_colors = colors[:num_categories]
category_to_base_color = {label: inner_colors[i] for i, label in enumerate(sorted_categories.keys())}

# Assign the solid base color to each virus entry
for item in virus_data:
    item['color'] = category_to_base_color.get(item['category'], (0.5, 0.5, 0.5)) # Default grey if category missing


# Sort the final data for plotting:
# 1. Primarily by category according to the pie chart's order.
# 2. Secondarily by the randomly sampled CDS length (descending) within each category.
sorted_virus_data_for_plot = sorted(
    virus_data,
    key=lambda item: (category_order_map.get(item['category'], 999), -item['random_length']) # Sort by random_length
)

# Extract final lists for plotting
plot_labels = [item['name'] for item in sorted_virus_data_for_plot]
plot_values = [item['random_length'] for item in sorted_virus_data_for_plot] # Use the random length
plot_colors = [item['color'] for item in sorted_virus_data_for_plot] # Use the assigned solid color

# --- Plotting ---

fig, ax = plt.subplots(figsize=(8, 5)) # Adjust figure size as needed

# Create the bar plot with solid colors per category
bars = ax.bar(plot_labels, plot_values, color=plot_colors)

# --- Customize Plot ---
ax.set_ylabel('Average CDS Length (bases)', fontsize=12)
ax.set_title('CDS Length per Virus (Grouped by Category)', fontsize=16, pad=20)
# Rotate x-axis labels by 45 degrees and align them to the right
ax.tick_params(axis='x', labelrotation=45, labelsize=9) # Adjust font size if needed
plt.xticks(ha='right') # Align rotated labels to the right for better spacing

ax.grid(axis='y', linestyle='--', alpha=0.7)

# --- Legend ---
# Legend section removed as requested

# Adjust layout automatically, giving some padding
plt.tight_layout(pad=1.5)

# Save the figure
output_filename = 'virus_random_cds_length_bar_plot_no_legend.png' # New filename
try:
    plt.savefig(output_filename, format='png', bbox_inches='tight', dpi=300)
    print(f"\nBar plot saved successfully as {output_filename}")
except Exception as e:
    print(f"\nError saving bar plot: {e}")

# Show the plot
plt.show()

# --- Print Summary Data ---
print("\n--- Randomly Sampled CDS Length per Virus (Sorted for Plot) ---")
# Note: The printed length is the one randomly sampled for *this run* of the script.
for item in sorted_virus_data_for_plot:
    print(f"- {item['name']} ({item['category']}): {item['random_length']} bases (Sampled from {item['cds_count']} CDS)")

print("\nAnalysis complete.")
