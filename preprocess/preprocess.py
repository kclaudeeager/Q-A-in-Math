import csv
import re
from tqdm import tqdm

def preprocess_data(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Question', 'Answer', 'Category'])

        current_category = ''
        for line in tqdm(infile):
            line = line.strip()
            if line:
                # Check if the line indicates a new category
                if line.startswith('train/'):
                    line = line.replace('train/', '')
                    current_category = line  # Update current category
                else:
                    # Split the question and answer at the last period (.)
                    parts = re.split(r'\s*[.?]\s*', line)
                    if len(parts) == 2:
                        question, answer = parts
                        question = question.strip()
                        answer = answer.strip()
                        # Write to CSV
                        csv_writer.writerow([question, answer, current_category])

def main():
    input_file = 'Dataset/questions_and_answers.txt'  # Replace with your input file name
    output_file = 'Dataset/preprocessed_data.csv'
    preprocess_data(input_file, output_file)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == '__main__':
    main()
