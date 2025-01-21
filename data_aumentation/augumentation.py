# from transformers import pipeline
# from tqdm import tqdm
# import pandas as pd

# # Load a pre-trained paraphrase generation model
# # paraphraser = pipeline("text2text-generation", model="t5-base")
# # paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
# from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# paraphraser = pipeline("text2text-generation", model="t5-base", device=0)
# # groq api key: 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


# def generate_paraphrases(question, num_paraphrases=5):
#     paraphrased_questions = []
#     for _ in range(num_paraphrases):
#         result = paraphraser(f"paraphrase: {question}", 
#                              max_length=100, 
#                              num_return_sequences=1,
#                              temperature=0.7,
#                              top_k=50,
#                              top_p=0.95)
#         paraphrased_questions.append(result[0]['generated_text'])
#     return paraphrased_questions


# def augment_dataset(dataset, num_paraphrases=5):
#     augmented_data = []
#     for _, row in tqdm(dataset.iterrows()):
#         question, answer, category = row['Question'], row['Answer'], row['Category']
#         paraphrases = generate_paraphrases(question, num_paraphrases)
#         for paraphrase in paraphrases:
#             augmented_data.append({
#                 "Question": paraphrase, 
#                 "Equation": question,  # Use the original question as the equation
#                 "Answer": answer, 
#                 "Category": category
#             })
#     return pd.DataFrame(augmented_data)

# def main():
#     input_file = 'Dataset/preprocessed_data.csv'
#     output_file = 'Dataset/augmented_data.csv'
#     data = pd.read_csv(input_file)
#     augmented_data = augment_dataset(data, num_paraphrases=5)
#     augmented_data.to_csv(output_file, index=False)
#     print(f"Augmented data saved to {output_file}")

# if __name__ == '__main__':
#     main()


import os
import pandas as pd
from groq import Groq
from tqdm import tqdm

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def generate_paraphrases(question, num_paraphrases=5):
    print(f"Paraphrasing {question}")
    paraphrased_questions = []
    for _ in range(num_paraphrases):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert mathematical language transformer. Rephrase mathematical questions while preserving their exact mathematical structure and meaning. Focus on syntactic variation without altering the core mathematical problem."
                    },
                    {
                        "role": "user", 
                        "content": f"Rephrase this mathematical question, ensuring the mathematical problem remains identical: {question}"
                    }
                ],
                model="gemma2-9b-it",
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                stop=None
            )
            paraphrase = chat_completion.choices[0].message.content.strip()
            
            # Basic validation to ensure paraphrase is not empty and different from original
            if paraphrase and paraphrase.lower() != question.lower():
                paraphrased_questions.append(paraphrase)
        except Exception as e:
            print(f"Error generating paraphrase: {e}")
    
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
