import openai, os
import argparse
import tqdm
import json
import ast 

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--api_key", required=True, help="OpenAI API key")
args = parser.parse_args()

openai.api_key = args.api_key

def annotate(transcript, Q, A):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """

    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an intelligent chatbot specialized in laparoscopic surgical topics. "
                            "You will be provided with the transcript of a surgical video, along with a question and answer. "
                            "The transcript contains the speech and instructions given by the doctor while performing the surgery. "
                            "These videos are for educational purposes for residents.\n\n"
                            "Your task is to classify the question into one of the following categories:\n"
                            "- **Observation**: Descriptions of surgical actions, organs, arteries, veins, etc., from the transcript.\n"
                            "- **Reason**: The reason or intention behind an observation (e.g., 'the reason for ... is to ...').\n"
                            "- **Plan**: Surgical actions that can be performed (e.g., 'after ..., we can ...').\n"
                            "- **Note**: Important notices related to an observation (e.g., 'when ..., note that ...').\n"
                            "- **Organs**: Organs involved in the surgical scenario.\n"
                            "- **Equipment**: Surgical equipment used in the scenario.\n\n"
                            "Return only the category name that best fits the question."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Based on the provided transcript, question, and answer, determine the appropriate category:\n\n"
                            f"**Transcript:** {transcript}\n"
                            f"**Question:** {Q}\n"
                            f"**Answer:** {A}\n\n"
                            "Provide only the category name."
                        ),
                    }
                ]

            )
        # Convert response to a Python dictionary.
        response_dict = completion["choices"][0]["message"]["content"]
        return response_dict

    except Exception as e:
        print(f"Error processing file {e}")
        return {}


def main():

    pred_path="/data/shared/gauravs/llapsa/surgical_tutor/all_qas/qual/train_val.json"
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_contents = json.load(file)

    os.makedirs("vqa_distribution")
    all_scr_files = os.listdir("vqa_distribution")

    for ind, pc  in enumerate(tqdm.tqdm(pred_contents, total=len(pred_contents))):
        try:
            qtn = pc["Q"]
            ans = pc["A"]
            text = pc["input_text"]
            cid = pc["counter_id"]

            if f"{cid}.txt" not in all_scr_files:
                response = annotate(text, qtn, ans)
                print(response)
                
                exit()
                with open(f"vqa_distribution/{cid}.txt", "w") as f:
                    f.write(f"{response}")
                
        except:
            didnot_work+=1
            print(f"{cid}.txt not working!")

if __name__ == "__main__":
    main()