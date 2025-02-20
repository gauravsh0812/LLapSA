import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
import tqdm


parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--openai_model", required=True, help="which openai model -- gpt-3.5-turbo or gpt-4o-mini")
parser.add_argument("--predicted_file_path", required=True, help="file containing the predictions from trained model (Inference output file)")
parser.add_argument("--output_dir", required=True, help="output directory")
args = parser.parse_args()

openai.api_key = args.api_key

def annotate(question, answer, pred,):
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                model=args.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an expert evaluator in surgical procedures. \
                            Your task is to assess the numerical accuracy, precision, and completeness of AI-generated responses to surgical video-based questions. \
                            You will determine whether the prediction is meaningfully correct (Yes/No) and assign a numerical accuracy score based on clinical relevance."
                            
                            "------"
                            "**Evaluation Criteria:**\n"
                            "Numerical Accuracy & Precision: The prediction must correctly match numerical values, measurements, time durations, and dosages relevant to the surgical context. Even small discrepancies impact accuracy.\n"
                            "Range Tolerance: Acceptable if within an appropriate clinical range (e.g., normal blood loss range or safe cautery settings). Significant deviations reduce the score.\n"
                            "Unit Consistency: Ensure numerical values are expressed in the correct units of measurement (e.g., mm vs. cm, seconds vs. minutes). Misuse of units lowers accuracy.\n"
                            "Calculation Verification: If the response involves calculations (e.g., fluid balance, dosage adjustments), verify correctness based on medical standards.\n"
                            "Clinical Applicability: The prediction should not only be numerically correct but also relevant to the surgical scenario. Meaningless or out-of-context numbers lower the score.\n"

                            "**Scoring System (1-5):**\n"
                            "**5 (Perfect)**: Fully accurate, complete, and medically precise.\n"
                            "**4 (Minor Omission)**: Mostly correct but **lacks minor surgical details**.\n"
                            "**3 (Partial Accuracy)**: Some inaccuracies **or missing key details**.\n"
                            "**2 (Major Inaccuracy)**: Contains significant errors **or misrepresents the procedure**.\n"
                            "**1 (Incorrect/Misleading)**: Entirely incorrect or misleading.\n\n"
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            return response_dict

        except Exception as e:
            print(f"Error processing file: {e}")


def main():

    """
    Main function to control the flow of the program.
    """

    pred_path=args.predicted_file_path
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)

    os.makedirs(f"{args.output_dir}", exist_ok=True)

    total_yes = 0
    total_no = 0
    total_score = 0
    length = 0

    didnot_work = 0

    all_scr_files = os.listdir(f"{args.output_dir}")

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):
        qtn = pc["question"]
        ans = pc["answer"]
        pred = pc["pred"]
        vid = pc["id"]
        
        if f"{vid}.txt" not in all_scr_files:
            try:
                response = annotate(qtn, pred, ans)
                y_n, scr = response["pred"], response['score']
                length+=1
                total_score += float(scr)
                if  "yes" in y_n.lower():
                    total_yes += 1
                elif "no" in y_n.lower():
                    total_no += 1

                with open(f"{args.output_dir}/{vid}.txt", "w") as f:
                    f.write(f"{vid} -- {y_n} -- {scr}")

            except:
                print(f"{vid}.txt not working!")
                didnot_work+=1
    
    average_score = total_score / length
    accuracy = total_yes / (total_yes + total_no)

    print("Yes count:", total_yes)
    print("No count:", total_no)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()
