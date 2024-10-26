import openai
import time
# from openai import OpenAI
import os
import argparse
import tqdm
import json
import ast
from multiprocessing.pool import Pool


# PUT here openai.api_key then work with it
# client = OpenAI()

def annotate(qtn, pred, ans):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """
    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content":
                                "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "-Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"                                   "- The predicted answer must be factually accurate and align with the video content.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the factual accuracy of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {qtn}\n"
                            f"Correct Answer: {ans}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
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

    pred_path="outputs/inference_output.json"
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)[x:y]

    scores = open(f"outputs/correctness/eval_1_correctness_scores_turbo3_{n}.lst", "w")
    len_scores = 0
    total_score = 0

    didnot_work = 0

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):
        try:
            #if ind % 100 == 0:
                #time.sleep(10)
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

