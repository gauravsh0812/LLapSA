import os, json, csv
import tqdm
import pandas as pd
from openai import OpenAI
import glob
import multiprocessing as mp
from openai import APIStatusError

# setting openapi_key to environ
os.environ["OPENAI_API_KEY"] = "<KEY>"
client = OpenAI()


def create_new_question(qtn):

    response = client.chat.completions.create(
    model="gpt-4o-mini",
        messages=[
        {"role": "system", 
         "content": "You are an intelligent English language chatbot. Your task is to reformulate the questions I provide, maintaining their original context. \
            Make sure that the context of the question must remain same. All you need to do is re-write it in a different way."
        },
        {"role": "user", 
         "content": f" Re-write this question without changing the original context. Here is the question: {qtn}"
        },
    ],

    max_tokens=1000,
    temperature=0.2,  # Lower temperature for more deterministic output
    top_p=0.9  # Lower top_p for less diverse output
    )

    return(response.choices[0].message.content)


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