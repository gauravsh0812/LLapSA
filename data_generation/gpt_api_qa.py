import openai, os
import argparse
import tqdm
import json
import ast 

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--input_file", required=True, help="Input json file containing all the obseravtion,Note,\
                                                            plan, reasons, equipemnets, and organ details.")
parser.add_argument("--output_json_file_path", required=True, help="output file path")
parser.add_argument("--xyn", required=True)
args = parser.parse_args()

openai.api_key = args.api_key

def annotate(text, details):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """

    try:
        # Compute the correctness score
        completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role":"system", 
                        "content": 
                            f"""You are an AI assistant specialized in surgical topics.
                            You are provided with a text description of a surgical video clip from a surgical lecture. Along with that, you will also be
                            provided the following information extracted from the text description: Obseravtion, plan or task or action details, 
                            reason behind the action, any special note to be aware of, information regrading equipments and organds. 
                            It is not necessray to provide all of these details. You will be provided with as much as details possible.
                            In some cases, you may have additional text (title, description). Unfortunately, you don't have access to the actual video.
                            Your task is to generate a Q&A pair or an answer to a given question about the video clip. The conversation
                            should proceed as though both the User and Assistant are viewing the video, while not referring to the text
                            information (title, description).

                            #-------------#
                            #INSTRUCTIONS#

                            Below are requirements for generating the questions and answers in the conversation:
                            - Avoid directly quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as
                            these may reveal the conversation is based on the text information, rather than the video clip itself. Focus on
                            the visual aspects of the video that can be inferred without the text information.
                            - Do not use phrases like "mentioned", "title", "description" in the conversation. Instead, refer to the
                            information as being "in the video."
                            There can be a few types of question as following: 
                                - reason which asks the reason of an action, 
                                - plan which ask a possible future step, 
                                - note which asks for something you should notice when perform some action, 
                                - detail which asks for more information about the observation
                                - details about the oragn and equipments used in the surgical scenario.

                            Generate a Q&A pair that you use the "statement" value to answer a question regarding the "observation".
                            Your reply should be in the following json format: {"q": "<the_question>", "a": "<the_answer>", "type": "<qa_type>"}, where \n
                            The <the_question> is the questions, <the_answers> is the asnwer to the question, and "<qa_type>" is the 
                            type of the questions which could be one of the following: obseravtion, note, plan, reason, equipment, or organ. 
                            """
                    },
                    {
                        "role": "user",
                        "content":
                            f"""Provide me the json dictionary of {"q": "<the_question>", "a": "<the_answer>", "type": "<qa_type>"}
                            for the given text and details associated with it. 
                            Here is the text description of the surgical video scenario: {text}. \n
                            Here are the important deatils associated with the provided text. It is provided in the form of the json dictionary: \n
                            {details}.
                            """
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

    """
    Main function to control the flow of the program.
    """
    
    xyn = args.xyn
    x,y,n = int(xyn.split("-"))
    
    output_json_file_path = args.output_json_file_path
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    if y!=-1:
        data = data[x:y]
    else:
        data = data[x:]

    all_responses = []
    count = 0
    didnot_work_count = 0
    for af in tqdm.tqdm(data, total=len(data)):
        try:
            video_id = af["video_id"]
            text = af["transcript"]
            details = {}
            if af["obseravtion"] != []:
                details["observation"] = af["observation"]
            if af["reason"] != []:
                details["reason"] = af["reason"]
            if af["plan"] != []:
                details["plan"] = af["plan"]
            if af["note"] != []:
                details["note"] = af["note"]
            if af["organs"] != []:
                details["organs"] = af["organs"]
            if af["equipments"] != []:
                details["equipments"] = af["equipments"]

            response = annotate(text, details)
            print(response)
            break
            response = ast.literal_eval(response)
            all_responses.append(response)

        except:
            didnot_work_count+=1
        exit()        
        count +=1

        if count % 500==0:
            # Write all responses to the JSON file
            print(f"writing {count} results...")
            output_file = open(output_json_file_path, "w")
            output_file.write(json.dumps(all_responses, indent=2))
            output_file.write('\n')  # Add a newline after the entire JSON object
    
    # Write all responses to the JSON file
    # output_file = open(output_json_file_path, "w")
    # output_file.write(json.dumps(all_responses, indent=2))
    # output_file.write('\n')  # Add a newline after the entire JSON object
    
    # print("failed file numbers: ", didnot_work_count)

if __name__ == "__main__":
    main()