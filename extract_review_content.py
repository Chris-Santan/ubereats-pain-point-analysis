import csv
import os

# Directories containing the filtered_reviews.csv files
input_dirs = [
    r'C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\results\topic_7_unauthorized_or_incorrect_tip_charges',
    r'C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\results\topic_24',
    r'C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\results\topic_2'
]

def extract_content_only(input_dir):
    """Extract only the content column from filtered_reviews.csv and save as review_content_only.csv"""
    input_file = os.path.join(input_dir, 'filtered_reviews.csv')
    output_file = os.path.join(input_dir, 'review_content_only.csv')
    
    if not os.path.exists(input_file):
        print(f"⚠ File not found: {input_file}")
        return
    
    # Read content column from input file
    review_contents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'content' in row:
                review_contents.append(row['content'])
    
    # Write to output CSV file
    if review_contents:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['content'])  # Header
            for num, content in enumerate(review_contents, 1):
                numbered_content = f"{num}. {content}"
                writer.writerow([numbered_content])
        
        print(f"✓ Saved {len(review_contents)} reviews to {output_file}")
    else:
        print(f"⚠ No content found in {input_file}")

# Process each directory
print("Extracting review content only...\n")
for i, input_dir in enumerate(input_dirs, 1):
    print(f"Processing {i}/3: {os.path.basename(input_dir)}")
    extract_content_only(input_dir)
    print()

print("Done!")
