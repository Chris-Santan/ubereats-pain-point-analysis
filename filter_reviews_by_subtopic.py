import csv
import os

# Base directory for data files
data_dir = os.path.join('data', 'deep_analysis')

# Configuration for each topic
configs = [
    {
        'input_file': os.path.join(data_dir, 'topic_7_subtopics.csv'),
        'subtopic_label': 'app, tip and order related issues',
        'output_dir': r'C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\results\topic_7_unauthorized_or_incorrect_tip_charges',
        'output_filename': 'filtered_reviews.csv'
    },
    {
        'input_file': os.path.join(data_dir, 'topic_24_subtopics.csv'),
        'subtopic_label': 'gift, card and gift card related issues',
        'output_dir': r'C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\results\topic_24',
        'output_filename': 'filtered_reviews.csv'
    },
    {
        'input_file': os.path.join(data_dir, 'topic_2_subtopics.csv'),
        'subtopic_label': 'codes, code and promo codes related issues',
        'output_dir': r'C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\results\topic_2',
        'output_filename': 'filtered_reviews.csv'
    }
]

def filter_and_save_reviews(input_file, subtopic_label, output_dir, output_filename):
    """Filter reviews by subtopic_label and save to CSV file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and filter entries
    matching_entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['subtopic_label'] == subtopic_label:
                matching_entries.append(row)
    
    # Write to output CSV file
    if matching_entries:
        output_path = os.path.join(output_dir, output_filename)
        fieldnames = matching_entries[0].keys()
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matching_entries)
        
        print(f"✓ Saved {len(matching_entries)} reviews to {output_path}")
    else:
        print(f"⚠ No reviews found with subtopic_label: '{subtopic_label}' in {input_file}")

# Process each configuration
print("Processing reviews by subtopic...\n")
for i, config in enumerate(configs, 1):
    print(f"Processing {i}/3: {os.path.basename(config['input_file'])}")
    filter_and_save_reviews(
        config['input_file'],
        config['subtopic_label'],
        config['output_dir'],
        config['output_filename']
    )
    print()

print("Done!")
