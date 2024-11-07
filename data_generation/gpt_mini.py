import os, json, csv
import yaml
import tqdm
import pandas as pd
from box import Box
from openai import OpenAI
import glob
import multiprocessing as mp
from openai import APIStatusError

with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

with open("config/gpt_config.yaml") as f:
    gpt_cfg = Box(yaml.safe_load(f))

root = cfg.dataset.path_to_data

# setting openapi_key to environ
os.environ["OPENAI_API_KEY"] = gpt_cfg.vsi_key
client = OpenAI()   


def getting_final_text_using_gpt4(raw_text):

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a professional endoscopic surgeon. And I am a newbie."
        },
        
        {"role": "user", "content": f" Re-write this raw text as a doctor in thid person voice such that it clearly and precisely \
                                    describes the text as a single paragraph. Here is the raw text: {raw_text}"
        },
    ],
    max_tokens=1000,
    temperature=0.2,  # Lower temperature for more deterministic output
    top_p=0.9  # Lower top_p for less diverse output
    )

    return(response.choices[0].message.content)


def getting_QA_using_gpt4(raw_text):

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional endoscopic surgeon."},
                {
                    "role": "user",
                    "content": f"The provided raw_text is the audio transcription of the video clip. From this, Create 5 question with the detailed answers from the provided raw text, \
                               that will help explain the context to a person in a simpler fashion. \
                               Ensure that you cover these points: at least one question describing the surgical scene, \
                                Atleast one question related to the organs involved in the surgery  ( if details are provided in the raw text) \
                                and, One question regarding the instruments ( if details are provided in the raw text) \
                                Rest you can create based on the information. \
                                Metion the question with 'Q:' and answers with 'A:'.\
                                Here is the raw text: {raw_text}"
                },
            ],
            max_tokens=1000,
            temperature=0.4,
            top_p=0.3
        )

        return(response.choices[0].message.content)
    
    except APIStatusError as e:
        print(f"API Status Error: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        return ""

def main(trns):
    try:
        vid = os.path.basename(trns).replace(".txt",".mp4")
        # print(vid)

        # Get the structured text
        full_text = "".join(open(trns, "rt", encoding="utf-8").readlines())
        final_text = getting_final_text_using_gpt4(full_text)
        # print(final_text)
        with open(f"/data/shared/gauravs/llapsa/structured_texts_from_gpt/{os.path.basename(trns)}","w") as f:
            f.write(final_text)
            f.close()

        # Get the QA
        qa = getting_QA_using_gpt4(final_text)
        qa_arr = qa.strip().split('\n')
        # print(qa_arr)
        
        qtns = []
        ans = []

        for _qa in qa_arr:
            if len(_qa.split()) > 2:
                l = _qa.replace("\n","").strip()
                if "Q:" in l:
                    qtns.append(l.replace("Q: ","").strip())
                elif "A:" in l:
                    ans.append(l.replace("A: ","").strip())

        return vid, qtns, ans
    
    except Exception as e:
        print(f"Error processing {trns}: {e}")
        return [], [], []


if __name__ == "__main__":
    
    n = 3
    N = 7000
    M = -1
    
    video_paths = []
    questions = []
    answers = []

    # Getting video_clips
    all_files = glob.glob("/data/shared/gauravs/llapsa/audio_transcriptions/*")[N:]
    for trns in tqdm.tqdm(all_files, total=len(all_files)):
        vid,qtns,ans = main(trns)

        for i in range(min(len(qtns), len(ans))):
            # print(vid)
            # print(qtns[i])
            # print(ans[i])
            video_paths.append(vid)
            questions.append(qtns[i])
            answers.append(ans[i])

    # Save the modified DataFrame 
    new_df = pd.DataFrame({
        "videos":video_paths,
        "questions":questions,
        "answers":answers
    })

    new_df.to_csv(f'/data/shared/gauravs/llapsa/final_qa_datatset_{n}.csv', index=False)

    # convert to json file 
    with open(f'/data/shared/gauravs/llapsa/final_qa_datatset_{n}.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    json_str = json.dumps(rows, indent=4)

    # Optionally, write the JSON string to a file
    with open(f'/data/shared/gauravs/llapsa/final_qa_datatset_{n}.json', 'w') as jsonfile:
        jsonfile.write(json_str)