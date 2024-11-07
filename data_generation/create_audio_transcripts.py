import os, json, csv
import yaml
import tqdm
import whisper
from moviepy.editor import VideoFileClip
import pandas as pd
from box import Box
from openai import OpenAI
import glob
import argparse

with open("config/config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

root = cfg.dataset.path_to_data

# whisper model
whisper_model = whisper.load_model("base")
# logging.getLogger("moviepy").setLevel(logging.ERROR)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process videos and generate QA data.')
    parser.add_argument('--generate_qa', default=False, 
                        action='store_true', help='Generate QA data using GPT')
    args = parser.parse_args()
    return args.generate_qa

def getting_final_text_using_gpt4(raw_text):

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a professional endoscopic surgeon. And I am a newbie."
        },
        
        {"role": "user", "content": f"Explain me the given text in such a way that it makes clear sense to me. \
                                    I want a single paragraph which is clean, accurate and precisely describing the given text. \
                                    The main idea here is to create a model that can take multiple input images defining a sequence \
                                    from a video and I want the model to produce a well rounded text. Re-describe this raw text as a \
                                    doctor in thid person voice: {raw_text}."
        },
    ],
    max_tokens=1000,
    temperature=0.2,  # Lower temperature for more deterministic output
    top_p=0.9  # Lower top_p for less diverse output
    )
    return(response.choices[0].message.content)


def getting_QA_using_gpt4(raw_text):

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a professional endoscopic surgeon."
        },
        
        {"role": "user", 
         "content": f"I want you to create 5 question with the detailed answers from the provided raw text, that will help explain the context to a person in a simpler fashion. Ensure that you cover these points: \
                    1) at least one question describing the text. \
                    2) Atleast one question related to the organs involved in the surgery  ( if details are provided in the raw text) \
                    3) One question regarding the instruments ( if details are provided in the raw text) \
                    Rest you can create based on the information. \
                    Metion the question with 'Q:' and answers with 'A:'.\
                    Here is the raw text: {raw_text}"
        },
    ],
    max_tokens=1000,
    temperature=0.4,  # Lower temperature for more deterministic output
    top_p=0.3  # Lower top_p for less diverse output
    )
    return(response.choices[0].message.content)

def transcribe_video_segment(video_filename, audio_filename):

    # Extract audio from the video segment
    print("video filename: ", video_filename)
    video = VideoFileClip(video_filename)
    video.audio.write_audiofile(audio_filename)
                                # verbose=False,
                                # logger=None)
    
    # Transcribe audio
    result = whisper_model.transcribe(audio_filename)
    os.remove(audio_filename)  # Remove the temporary audio file
    return result['segments']

if __name__ == "__main__":
    
    generate_qa = parse_arguments()

    n = 0
    N = 0
    M = 1000

    if not generate_qa:
        os.makedirs("/data/shared/gauravs/llapsa/videos_clips", exist_ok=True)
        os.makedirs("/data/shared/gauravs/llapsa/audio_clips", exist_ok=True)
        os.makedirs("/data/shared/gauravs/llapsa/audio_transcriptions", exist_ok=True)
        error_file = open("/data/shared/gauravs/llapsa/corrupt_video_clips.lst","w")
    else:
        print("First generatet the video clips, audio clips, respectively.")
        exit()

    df = pd.read_csv(f"data/video_list_LlapSA.csv")

    # shuffling
    df = df.sample(frac=1).reset_index(drop=True)
    
    # lists
    video_paths = []
    questions = []
    answers = []

    # getting video_clips
    for _,row in df.iterrows():
        vid = row["serial_no"]
        for sec in ["45sec","60sec","90sec"]:
            files = glob.glob(os.path.join("/data/shared/gauravs/llapsa/videos_clips/",
                                            f"{vid}_{sec}_*.mp4"))
            for f in files:
                    # getting the audios
                    name = os.path.basename(f)
                    part = name.split("_")[-1].split(".")[0]
                    audiofile_name = f"/data/shared/gauravs/llapsa/audio_clips/{vid}_{sec}_{part}.wav"
                    audiofile_path = f"/data/shared/gauravs/llapsa/audio_transcriptions/{vid}_{sec}_{part}.txt"
                    
                    if not generate_qa:
                        if not os.path.exists(audiofile_path):
                            try:
                                audio_data = transcribe_video_segment(f, audiofile_name)
                                texts = []
                                for ad in audio_data:
                                    texts.append(ad["text"])
                                
                                full_text = " ".join(texts)
                                
                                with open(os.path.join(audiofile_path),"w") as _wav:
                                    _wav.write(full_text)
                            except:
                                print(f"=================  {name} not working!! =====================")
                                error_file.write(f+"\n")
                        
                    # get the structured text
                    if generate_qa:
                        print("getting staructured gpt output...")
                        full_text = open(audiofile_path,"r")
                        final_text = getting_final_text_using_gpt4(full_text)

                        # get the QA
                        qa = getting_QA_using_gpt4(final_text)
                        qa_arr = qa.strip().split('\n')

                        qtns = []
                        ans = []

                        for _qa in qa_arr:
                            if len(_qa.split()) > 2:
                                l = _qa.replace("\n","").strip()
                                if "Q:" in l:
                                    qtns.append(l.replace("Q: ","").strip())
                                elif "A:" in l:
                                    ans.append(l.replace("A: ","").strip())

                        for i in range(len(qtns)):
                            video_paths.append(f)
                            questions.append(qtns[i])
                            answers.append(ans[i])

    if generate_qa:
        print("saving new df...")
        # Save the modified DataFrame 
        new_df = pd.DataFrame(
            {
                "videos":video_paths,
                "questions":questions,
                "answers":answers
            }
        )

        new_df.to_csv(f'/data/shared/gauravs/llapsa/final_qa_datatset_{n}.csv', index=False)

        # convert to json file 
        with open(f'/data/shared/gauravs/llapsa/final_qa_datatset_{n}.csv', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        json_str = json.dumps(rows, indent=4)

        # Optionally, write the JSON string to a file
        with open(f'/data/shared/gauravs/llapsa/final_qa_datatset_{n}.json', 'w') as jsonfile:
            jsonfile.write(json_str)