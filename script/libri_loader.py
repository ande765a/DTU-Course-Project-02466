#data - dev clean - ydre mappe - indre mappe - vælg trans
import os

path = '/Users/Gabi/Desktop/Fagprojekt/DTU-Course-Project-02466/data/dev-clean'
output_path = '/Users/Gabi/Desktop/Fagprojekt/DTU-Course-Project-02466/data/dev-clean-synthesize'

for speaker_id in os.listdir(path):
	speaker_path = os.path.join(path, speaker_id)
      
	for chapter_id in os.listdir(speaker_path):
		chapter_path = os.path.join(speaker_path, chapter_id)

		with open(os.path.join(chapter_path,f"{speaker_id}-{chapter_id}.trans.txt"),"r") as file:
			for line in file.readlines():
				audio_filename, text = line.split(" ",1)
				outputfile_path = os.path.join(output_path,speaker_id, chapter_id, f"{audio_filename}.flac")

			


