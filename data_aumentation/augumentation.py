from transformers import pipeline
from tqdm import tqdm
import pandas as pd

# Load a pre-trained paraphrase generation model
paraphraser = pipeline("text2text-generation", model="t5-base")

def generate_paraphrases(question, num_paraphrases=5):
    paraphrased_questions = []
    for _ in range(num_paraphrases):
        result = paraphraser(f"paraphrase: {question}", max_length=100, num_return_sequences=1)
        paraphrased_questions.append(result[0]['generated_text'])
    return paraphrased_questions

def augment_dataset(dataset, num_paraphrases=5):
    augmented_data = []
    for _, row in tqdm(dataset.iterrows()):
        question, answer, category = row['Question'], row['Answer'], row['Category']
        paraphrases = generate_paraphrases(question, num_paraphrases)
        for paraphrase in paraphrases:
            augmented_data.append({
                "Question": paraphrase, 
                "Equation": question,  # Use the original question as the equation
                "Answer": answer, 
                "Category": category
            })
    return pd.DataFrame(augmented_data)

def main():
    input_file = 'Dataset/preprocessed_data.csv'
    output_file = 'Dataset/augmented_data.csv'
    data = pd.read_csv(input_file)
    augmented_data = augment_dataset(data, num_paraphrases=5)
    augmented_data.to_csv(output_file, index=False)
    print(f"Augmented data saved to {output_file}")

if __name__ == '__main__':
    main()
