import openai, os
import argparse
import tqdm
import json
import ast

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--openai_model", required=True, help="which openai model -- gpt-3.5-turbo or gpt-4o-mini")
parser.add_argument("--predicted_file_path", required=True, help="file containing the predictions from trained model (Inference output file)")
parser.add_argument("--output_dir", required=True, help="output directory")
args = parser.parse_args()

openai.api_key = args.api_key

def annotate(qtn, pred, ans):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """

    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model=args.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the consistency of generative outputs for similar video-based question-answer pairs. "
                            "You will be given two very similar questions, a common answer common to both the questions and predicted answers for the two questions ."
                            "Your task is to compare the predicted answers for two very similar question, with a common correct answer and determine if they are consistent. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the consistency between the two predicted answers and the correct answer. Both predicted answers should correspond to the correct answer and to each other, and should not contain any contradictions or significant differences in the conveyed information.\n"
                            "- Both predicted answers must be consistent with each other and the correct answer, in terms of the information they provide about the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches, but only if they maintain the consistency in the conveyed information.\n"
                            "- Evaluate the consistency of the two predicted answers compared to the correct answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question 1: {question1}\n"
                            f"Question 2: {question2}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer to Question 1: {pred1}\n"
                            f"Predicted Answer to Question 2: {pred2}\n\n"
                            "Provide your evaluation only as a consistency score where the consistency score is an integer value between 0 and 5, with 5 indicating the highest level of consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the consistency score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {''score': 4.8}."
                    }
                ]
            )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)
        return response_dict

    except Exception as e:
        print(f"Error processing file {e}")


def main():

    """
    Main function to control the flow of the program.
    """
    
    x, y = 6000,-1
    n = 3
    
    pred_path=args.predicted_file_path
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)[x:y]

    os.makedirs(f"{args.output_dir}", exist_ok=True)
    os.makedirs(f"{args.output_dir}/detailed_orientation", exist_ok=True)

    mtype = "turbo3" if args.openai_model == "gpt-3.5-turbo" else "mini4o"
    scores = open(f"{args.output_dir}/detailed_orientation/detailed_orientation_scores_{mtype}_{n}.lst", "w")
    len_scores = 0
    total_score = 0

    didnot_work = 0

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):
        try:
            qtn = pc["q"]
            ans = pc["a"]
            pred = pc["pred"]
            vid = pc["video_id"]

            response = annotate(qtn, pred, ans)
            scr = response['score']
            len_scores+=1
            total_score += scr

            scores.write(f"{vid} \t score: {scr} \n")

        except:
            didnot_work+=1

    average_score = total_score / len_scores

    print("Average score for correctness:", average_score)
    print("didnot work: ", didnot_work)

if __name__ == "__main__":
    main()