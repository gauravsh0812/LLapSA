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


def annotate(question, answer, pred,):
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                model=args.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
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
                if y_n.lower() == "yes":
                    total_yes += 1
                elif y_n.lower() == "no":
                    total_no += 1

                with open(f"{args.output_dir}/{vid}.txt", "w") as f:
                    f.write(f"{vid} -- {y_n} -- {scr}")

            except:
                print(f"{vid}.txt not working!")
                didnot_work+=1





if __name__ == "__main__":
    main()
