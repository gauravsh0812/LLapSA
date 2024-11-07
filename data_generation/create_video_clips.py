import os
import subprocess
import pandas as pd

error_file = open("/data/shared/gauravs/llapsa/corrupt_videos.lst","w")

def get_video_duration(input_file):
    """
    Get the duration of the video in seconds.
    
    Args:
    input_file (str): Path to the input video file.
    
    Returns:
    float: Duration of the video in seconds.
    """
    command = ['ffmpeg', '-i', input_file, '-hide_banner']
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for line in result.stderr.splitlines():
        if 'Duration' in line:
            time_str = line.split(' ')[3].split(',')[0]
            h, m, s = time_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)
    return None

def split_video(input_file, output_dir, start_time_sec, end_time_sec, duration, clip_type, serial_no):
    """
    Splits a segment of a video into multiple clips of the specified duration and saves them in the specified directory.
    
    Args:
    input_file (str): Path to the input video file.
    output_dir (str): Directory where the output clips will be saved.
    start_time (float): Starting time (in sec) from where the segment of interest begins.
    end_time (float): Ending time (in sec) of the segment of interest.
    duration (int): Duration of each clip in seconds.
    clip_type (str): The label for the type of clip (e.g., '90sec', '60sec', '45sec').
    serial_no (str): Unique identifier for the video file.
    """

    # Get the base name of the video (without extension)
    base_name = f"{serial_no}_{clip_type}"

    clip_index = 1
    current_time = start_time_sec

    while current_time < end_time_sec:
        # Calculate clip length, adjust if the remaining duration is less than the specified duration
        clip_length = min(float(duration), float(end_time_sec) - float(current_time))
        
        # Output file for each clip
        output_file = os.path.join(output_dir, f"{base_name}_part{clip_index}.mp4")
        
        command = [
            'ffmpeg', '-i', input_file, 
            '-ss', str(current_time), 
            '-t', str(clip_length), 
            '-c', 'copy', output_file
        ]
        
        try:
            print(f"Creating {str(clip_type)} clip {str(clip_index)} \
                  from {str(current_time)} to {str(current_time) + str(clip_length)} seconds...")
                  
            subprocess.run(command, check=True)
            print(f"Saved {clip_type} clip as {output_file}")
        except subprocess.CalledProcessError as e:
            error_file.write(input_file + "\n")
            print(f"Failed to process clip {clip_index} of {clip_type}: {e}")
        
        current_time += duration
        clip_index += 1

def process_videos(df, video_dir, output_dir):
    
    # Define clip durations in seconds
    durations = {
        '90sec': 90,
        '60sec': 60,
        '45sec': 45
    }

    # Iterate over the rows in the DataFrame
    for _, row in df.iterrows():
        begin = row["begin_time_stamp_in_min"]
        end = row["end_time_stamp_in_min"]
        serial_no = row["serial_no"]

        barr = begin.split(",")
        if len(barr) == 2:
            begin = int(barr[0])*60 + int(barr[1])
        elif len(barr) == 3:
            begin = int(barr[0])*3600 + int(barr[1])*60 + int(barr[0])

        earr = end.split(",")
        if len(earr) == 2:
            end = int(earr[0])*60 + int(earr[1])
        elif len(earr) == 3:
            end = int(earr[0])*3600 + int(earr[1])*60 + int(earr[0])

        print("begin, end: ", begin, end)

        # Input video file for the current row
        input_file = os.path.join(video_dir, f"{serial_no}.mp4")
        
        # Check if the video file exists
        if not os.path.exists(input_file):
            print(f"Video file {input_file} does not exist. Skipping.")
            continue
        
        # Split the video segment into clips for each duration (90s, 60s, 45s)
        for clip_type, duration in durations.items():
            split_video(input_file, output_dir, begin, end, duration, clip_type, serial_no)

if __name__ == "__main__":
    # Path to the directory containing the videos
    video_dir = "/data/shared/gauravs/llapsa/videos"  # Update this to the directory containing your videos
    # Path to the directory where the output clips will be saved
    output_dir = "/data/shared/gauravs/llapsa/videos_clips"  # Update this to where you want to save the clips
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv("data/video_list_LlapSA.csv")

    # Process the videos
    process_videos(df, video_dir, output_dir)
