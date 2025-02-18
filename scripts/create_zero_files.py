import json
import ast
import tqdm
import argparse

parser = argparse.ArgumentParser(description="creting zero-shot")
parser.add_argument("--api_key", required=True,)
args = parser.parse_args()
openai.api_key = args.api_key

def annotate_current(text):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """

    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an intelligent chatbot designed to create zero-shot question-answer pairs. "
                                "Your task is to generate 3 question-answer pairs from the given text based on these instructions:\n\n"
                                "## INSTRUCTIONS:\n"
                                "- Create one question-answer pair where the answer is 'Yes'.\n"
                                "- Create one question-answer pair where the answer is 'No'.\n"
                                "- Create one question-answer pair where the answer is a single word, but the question must be a proper, grammatically correct question.\n"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "Please generate three question-answer pairs from the provided text:\n\n"
                                f"text: {text}\n\n"
                                "The output should be a list of dictionaries, each containing a question ('Q') and its corresponding answer ('A').\n"
                                "For example:\n"
                                "[{'Q': 'question1', 'A': 'answer1'}, {'Q': 'question2', 'A': 'answer2'}, {'Q': 'question3', 'A': 'answer3'}]"
                            )
                        }
                    ]
                )
        # Convert response to a Python dictionary.
        response_message = completion["choices"][0]["message"]["content"]
        response_dict = ast.literal_eval(response_message)
        return response_dict

    except Exception as e:
        print(f"Error processing file {e}")

def annotate_future(text):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """

    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
            model=args.openai_model,
            messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an intelligent chatbot designed to create zero-shot question-answer pairs. "
                                "Your task is to generate 3 question-answer pairs from the given text based on these instructions:\n\n"
                                "## INSTRUCTIONS:\n"
                                "- the question should be focussing on the future such that that sound like a person is inquiring about the next steps or plans from a surgeon \
                                   drawing insights from previously observed actions, events, or videos."
                                "- Create one question-answer pair where the answer is 'Yes'.\n"
                                "- Create one question-answer pair where the answer is 'No'.\n"
                                "- Create one question-answer pair where the answer is a single word, but the question must be a proper, grammatically correct question.\n"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "Please generate three question-answer pairs from the provided text:\n\n"
                                f"text: {text}\n\n"
                                "The output should be a list of dictionaries, each containing a question ('Q') and its corresponding answer ('A').\n"
                                "For example:\n"
                                "[{'Q': 'question1', 'A': 'answer1'}, {'Q': 'question2', 'A': 'answer2'}, {'Q': 'question3', 'A': 'answer3'}]"
                            )
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

    #q_pred_path="zero_shot_questions.json"
    #a_pred_path="zero_shot_answers.json"

    with open("/data/shared/gauravs/llapsa/surgical_tutor/all_qas/qual/test.json", "rb") as f:
        data = json.load(f)

    qtns = []
    ans = []
    for item in data:
        assert len(qtns)==len(ans)
        if len(qtns) < 2000:
            id = item["id"]
            text = item["input_text"]
            try:
                if "60sec" in id:
                    response = annotate_current(text)
                elif "45sec" in id:
                    response = annotate_future(text)

                for i,r in enumerate(response):
                    qtns.append(
                        {"video_name":id,
                        "question":r["Q"],
                        "question_id":f"q_{id}_{i}"}
                    )
                    ans.append(
                        {"video_name":id,
                        "answer":r["A"],
                        "question_id":f"q_{id}_{i}"
                        }
                    )

            except:
                pass
            
        else:
            break

        if len(qtns) % 300 == 0:
            print("writing QA....")
            with open("/data/shared/gauravs/llapsa/surgical_tutor/all_qas/zero/zero_shot_questions.json", "w") as file:
                json.dump(qtns, file, indent=2)
            with open("/data/shared/gauravs/llapsa/surgical_tutor/all_qas/zero/zero_shot_answers.json", "w") as file:
                json.dump(ans, file, indent=2)

    with open("/data/shared/gauravs/llapsa/surgical_tutor/all_qas/zero/zero_shot_questions.json", "w") as file:
        json.dump(qtns, file, indent=2)
    with open("/data/shared/gauravs/llapsa/surgical_tutor/all_qas/zero/zero_shot_answers.json", "w") as file:
        json.dump(ans, file, indent=2)

if __name__ == "__main__":
    main()