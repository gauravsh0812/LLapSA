import openai, os
import argparse
import tqdm
import json
import ast 

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--input_folder", required=True, help="Input folder conatining all audio transcripts")
parser.add_argument("--output_json_file_path", required=True, help="output directory to save")
args = parser.parse_args()

openai.api_key = args.api_key

def annotate(transcript):

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
                        "role": "system",
                        "content":
                            """
                            You are an intelligent chatbot specialized in the various laparscopic surgical topics. 
                            You will be provided with the transript of the surgical video. The transcript is the speech and instructions given by doctor while performing the surgey.
                            These videos are for the educational purposes for residents.
                            Your task is to annotate the data in a structured way, extract the meaningful visual descriptions from the transcript.
                            Your reply should follow the following json format:
                            { 
                                'observation': the observation includes the descriptions to surgical actions or organs, arteries, veins, etc. from the transcript,
                                'reason': the reason or intention behind the observation if any foe example 'the reason of … is to …',
                                'plan': the surgical actions that can be performed for example 'after …, we can …',
                                'note': notice about the observation for example 'when …, note that …',
                                'organs': organs involved in the surgerical scenario,
                                'equipments': surgical euipments are used in the scenario,
                            }

                            ------
                            ##INSTRUCTIONS: 
                            Below are requirements for the annotations:
                            - Annotations may contain multiple observations and its corresponding reasons, plans, or notes.
                            - Always use list to wrap the content even if there is only 1 observation.
                            - Observation must only be descriptions to visible objects, or actions.
                            - Return an empty list if there is no descriptions to visible object, action, equipments, or organs in the transcript.
                            """
                    },
                    {
                        "role": "user",
                        "content":
                            f"Provide me the list of json dictionary for the given video transcription: \n\n {transcript}"
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
    
    x, y = 12000,14000
    n = 0
    
    all_files = os.listdir(args.input_folder)[x:]
    output_json_file_path = args.output_json_file_path
    
    didnot_work = open("failed_api_4.lst","w")
    didnot_work_count = 0

    all_responses = []
    count = 0
    for af in tqdm.tqdm(all_files, total=len(all_files)):
        try:
            transcript = open(f"{args.input_folder}/{af}").readlines()
            transcript = " ".join(transcript)
            response = annotate(transcript)
            if "```json" in response:
                response = response.replace("```json", "").replace("```", "")
            response = ast.literal_eval(response)
            if isinstance(response, list):
                response = response[0]            
            response["video_id"] = af
            response["transcript"] = transcript
            all_responses.append(response)
        except:
            didnot_work.write(af + "\n")
            didnot_work_count+=1
        
        count +=1

        if count % 500==0:
            # Write all responses to the JSON file
            print(f"writing {count} results...")
            output_file = open(output_json_file_path, "w")
            output_file.write(json.dumps(all_responses, indent=2))
            output_file.write('\n')  # Add a newline after the entire JSON object
    
    # Write all responses to the JSON file
    output_file = open(output_json_file_path, "w")
    output_file.write(json.dumps(all_responses, indent=2))
    output_file.write('\n')  # Add a newline after the entire JSON object
    
    print("failed file numbers: ", didnot_work_count)

if __name__ == "__main__":
    main()