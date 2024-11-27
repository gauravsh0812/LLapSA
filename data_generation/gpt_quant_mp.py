import openai, os
import argparse
import tqdm
import json
import ast 

parser = argparse.ArgumentParser(description="Training")

parser.add_argument("--api_key", required=True, help="OpenAI API key")
parser.add_argument("--input_file", required=True, help="Input folder conatining all audio transcripts")
parser.add_argument("--output_json_file_path", required=True, help="output directory to save")
parser.add_argument("--xy", required=True)
args = parser.parse_args()

openai.api_key = args.api_key

def annotate(Question, Answer):

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
                        "content": "You are an advanced AI language model specializing in generating concise, one-word or Yes/No answers for zero-shot evaluation. \
                                    Based on the provided QA pair, generate related question-answer pairs with straightforward answers. \
                                    The answers must be factual, unambiguous, and limited to one word answers. Also, include the Yes/No questions answers. \
                                    Focus on diversity in the questions while keeping the answers succinct."
                    },
                    {
                        "role": "user",
                        "content": f"Here is a QA pair for reference:\n\n {Question} \n {Answer}.\n \
                                    Using this as a base, generate at least 3 related QA pairs with Yes/No or one-word answers. \
                                    Keep the answers concise and appropriate for quantitative evaluation. \n \
                                    The output should be two separate lists, one for questions and another for corresponding answers."
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
    
    x,y = args.xy.split("-")
    
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    sorted_data = sorted(data, key=lambda x: x['video_id'])
    f = sorted_data[int(x):int(y)]

    output_json_file_path = args.output_json_file_path
    
    all_responses = []
    count = 0
    for af in tqdm.tqdm(f, total=len(f)):
        try:
            q = af["q"]
            a = af["a"]
            video_id = af["video_id"]
            response = annotate(q,a)
            
            # if "```json" in response:
            #     response = response.replace("```json", "").replace("```", "")
            # response = ast.literal_eval(response)
            # if isinstance(response, list):
            #     response = response[0]            
            
            # for r in response:
            print(len(response))
            print(response)
            all_responses.append(response)

            # if q == "What equipments are used in the surgical video?":


            
        except:
            pass
        
        count +=1

        if count % 500==0:
            # Write all responses to the JSON file
            print(f"writing {count} results...")
            output_file = open(output_json_file_path, "w")
            output_file.write(json.dumps(all_responses, indent=2))
            output_file.write('\n')  # Add a newline after the entire JSON object
    
    exit()
    # Write all responses to the JSON file
    output_file = open(output_json_file_path, "w")
    output_file.write(json.dumps(all_responses, indent=2))
    output_file.write('\n')  # Add a newline after the entire JSON object
    
    print("failed file numbers: ", didnot_work_count)

if __name__ == "__main__":
    main()