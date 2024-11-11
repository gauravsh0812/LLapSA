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

def msg_system(key):
    messages_system= {
        "observation": "You are an AI assistant specialized in surgical topics. \
                        You are provided with a text description/transcription of a surgical video clip from a surgical lecture. \
                        Along with that, you will be provided with the 'observations' made by the medical professional. Unfortunately, you don't have access to the actual video. \
                        Your task is to generate a Q&A pair or an answer to a given question about the video clip based on  the information you can gather from the text description and observation. \
                        The conversation should proceed as though both the User and Assistant are viewing the video, while not referring to the text information such as title, description.",
    
        "reason":   "You are an AI assistant specialized in surgical topics. \
                        You are provided with a text description/transcription of a surgical video clip from a surgical lecture. \
                        Along with that, you will be provided with the 'reasons' of the actions mentioned in the text description. Unfortunately, you don't have access to the actual video. \
                        Your task is to generate a Q&A pair or an answer to a given question about the video clip based on  the information you can gather from the text description and reasons. \
                        The conversation should proceed as though both the User and Assistant are viewing the video, while not referring to the text information such as title, description.",
        
        "plan":     "You are an AI assistant specialized in surgical topics. \
                        You are provided with a text description/transcription of a surgical video clip from a surgical lecture. \
                        Along with that, you will be provided with the 'plan' of the future actions based on information provided in the text description. Unfortunately, you don't have access to the actual video. \
                        Your task is to generate a Q&A pair or an answer to a given question about the video clip based on  the information you can gather from the text description and plan. \
                        The conversation should proceed as though both the User and Assistant are viewing the video, while not referring to the text information such as title, description.",

        "note":     "You are an AI assistant specialized in surgical topics. \
                        You are provided with a text description/transcription of a surgical video clip from a surgical lecture. \
                        Along with that, you will be provided with the 'note' regarding some action taken by the medical professional explained in the text description. Unfortunately, you don't have access to the actual video. \
                        Your task is to generate a Q&A pair or an answer to a given question about the video clip based on  the information you can gather from the text description and plan. \
                        The conversation should proceed as though both the User and Assistant are viewing the video, while not referring to the text information such as title, description.",
            
        "description": "You are an AI assistant specialized in surgical topics. Using the provided details, create a description of the surgical procedure. \
                        Include the sequence of actions, observations made during each step, the overall plan guiding the surgery, and the reasons behind each maneuver. \
                        Ensure that the description highlights the surgeon's approach to carefully cleaning, dissecting, and identifying critical anatomical structures while maintaining precision and minimizing disturbance. \n \
                        The details that will be provided by the User are: \
                            Transcript: The text description of the surgical video. \
                            Observations: The observations made form  the text description. \
                            Plan: The plans for the next step or future actions mentioned by surgeon in the video. \
                            Reason: The reasons provided by the surgeon for the actions/steps taken in the video during the surgery. \
                            Note: Any important point mentioned by the surgeon which should be taken care of while performing the surgery. \n \
                        Using these elements, create a question-answer pair to provide a cohesive and informative description of the surgery."
            
    }
    return messages_system[key]

def msg_user(key, text, detail):

    if key == "observation":
        prompt = f"Provide me the json dictionary of a Q&A pair,  for the given text and obseravtion associated with it. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the observation made by the medical professional: {detail}",

    elif key == "reason": 
        prompt = f"Provide me the json dictionary of question-answer pair,  for the given text and the reason provided. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the reason provided by the medical professional for the action: {detail}",

    elif key == "plan":     
        prompt = f"Provide me the json dictionary of question-answer pair,  for the given text and plan for the next steps. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the plan provided by the medical professional for the future actions: {detail}",
    
    elif key == "note":     
        prompt = f"Provide me the json dictionary of question-answer pair,  for the given text and important point mentioned by surgeon. \
                    Here is the text description of the surgical video scenario: {text} \n\n \
                    And here is the important note mentioned by the medical professional: {detail}",
    
    elif key == "description": 
        prompt = f"Based on the information provided down below, give the comprehensove description of the surgery. \
                        provide  the final reponse in the form of a question-answer pair. \
                        Here is the text description of the surgical video scenario: {text} \n\n \
                        Along with the text description, here are the list of the observations, reasons, plan, and notes related to the surgical video, mentioned by the surgeon. \
                        Observations: {detail['observations']} \n \
                        Reasons: {detail['reasons']} \n \
                        Plans: {detail['plans']} \n \
                        Notes: {detail['notes']}"
    
    return prompt

def annotate(key, text, detail):

    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness
    """

    # try:
        # Compute the correctness score
    system_text = msg_system(key)
    user_text = msg_user(key, text, detail)
    completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role":"system", 
                        "content": 
                            f"""
                            {system_text}

                            #-------------#
                            #INSTRUCTIONS#

                            Below are requirements for generating the questions and answers in the conversation:
                            - Avoid directly quoting or referring to specific facts, terms, abbreviations, dates, numbers, or names, as
                            these may reveal the conversation is based on the text information, rather than the video clip itself. Focus on
                            the visual aspects of the video that can be inferred without the text information.
                            - Do not use phrases like "mentioned", "title", "description" in the conversation. Instead, refer to the
                            information as being "in the video."

                            Your reply should be in the following json dictionary format where the key will be the question and the value will be the answer. 
                            """
                    },
                    {
                        "role": "user",
                        "content":
                                f"{user_text}"
                    }
                ]
            )
    # Convert response to a Python dictionary.
    response_dict = completion["choices"][0]["message"]["content"]
    return response_dict

    # except Exception as e:
    #     print(f"Error processing file {e}")
    #     return {}


def main():

    """
    Main function to control the flow of the program.
    """
    
    xyn = args.xyn
    x,y,n = xyn.split("-")
    
    output_json_file_path = args.output_json_file_path
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    if y!=-1:
        data = data[int(x):int(y)]
    else:
        data = data[int(x):]

    def get_response(key, text, o):
        response = annotate(key, text, o)
        # print(response)
        response = ast.literal_eval(response)
        return response
    
    all_responses = []
    count = 0
    didnot_work_count = 0
    for af in tqdm.tqdm(data, total=len(data)):
        # try:
        video_id = af["video_id"]
        text = af["transcript"]

        # print("text: ", text + "\n")

        obs = af["observation"]
        rsn = af["reason"]
        pln = af["plan"]
        nt = af["note"]
        ogn = af["organs"]
        eqp = af["equipments"]

        # getting QA
        for o in obs:        
            print("obs: ", o)    
            response = get_response("observation", text,o)
            print(response)
            all_responses.append(response)
        for r in rsn:
            print("rsn: ", r)
            response = get_response("observation", text,r)
            print(response)
            all_responses.append(response)
        for p in pln:
            print("pln: ", p)
            response = get_response("observation", text,p)
            print(response)
            all_responses.append(response)
        for n in nt:
            print("nt: ", n)
            response = get_response("observation", text,n)
            print(response)
            all_responses.append(response)

        # final description QA
        details = {}
        details['observations'] = obs
        details['reasons'] = rsn
        details['plans'] = pln
        details["notes"] = nt
        print("detail: ", details)
        response = get_response("description", text, details)
        print(response)
        print(' ')
        all_responses.append(response)

        print(details)

        # except:
        #     didnot_work_count+=1
        # exit()        
        count +=1

    print("all_responses: \n", all_responses)

        # if count % 500==0:
        #     # Write all responses to the JSON file
        #     print(f"writing {count} results...")
        #     output_file = open(output_json_file_path, "w")
        #     output_file.write(json.dumps(all_responses, indent=2))
        #     output_file.write('\n')  # Add a newline after the entire JSON object
    
    # Write all responses to the JSON file
    # output_file = open(output_json_file_path, "w")
    # output_file.write(json.dumps(all_responses, indent=2))
    # output_file.write('\n')  # Add a newline after the entire JSON object
    
    # print("failed file numbers: ", didnot_work_count)

if __name__ == "__main__":
    main()