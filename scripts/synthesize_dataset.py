import sys
import os

if __name__ == "__main__":
  _, path, output_path = sys.argv

  for speaker_id in os.listdir(path):
    speaker_path = os.path.join(path, speaker_id)

    for chapter_id in os.listdir(speaker_path):
      chapter_path = os.path.join(speaker_path, chapter_id)
      transcript_filename = f"{speaker_id}-{chapter_id}.trans.txt"
      transcript_path = os.path.join(chapter_path, transcript_filename)

      with open(transcript_path, "r") as file:
        for line in file.readlines():
          audio_name, text = line.split(" ", 1)
          audio_filename = f"{audio_name}.flac"
          audio_output_dir = os.path.join(output_path, speaker_id, chapter_id)
          audio_output_path = os.path.join(audio_output_dir, audio_filename)

          # Generate the actual audio
