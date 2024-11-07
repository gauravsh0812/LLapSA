import subprocess
import os
import pandas as pd

def download_youtube_video(video_url, output_filename,):
    command = ['yt-dlp', '--cookies', f"data/cookies.txt", '-o', output_filename, video_url]
    subprocess.run(command, check=True)
    print(f"Video downloaded as {output_filename}")

if __name__ == "__main__":

    root = "data"
    df = pd.read_csv(f"{root}/video_list_LlapSA.csv")[359:360]

    os.makedirs(f"{root}/videos_2", exist_ok=True)

    for i,row in df.iterrows():
        try:
            sn = row["serial_no"]
            vid = row["video_link"]
            video_filename = f"{root}/videos_2/{sn}.mp4"

            # Download video
            if not os.path.exists(video_filename):
                download_youtube_video(vid, video_filename)
            
            print(f"{sn} has been downloaded.")
        except:
            break                                    