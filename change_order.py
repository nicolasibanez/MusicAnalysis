from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TALB
import pandas as pd

def set_album(file_path, album_number):
    audio = MP3(file_path, ID3=ID3)

    if audio.tags is None:
        audio.add_tags()

    # Set new album number
    audio.tags.add(TALB(encoding=3, text=str(album_number)))
    audio.save()

    print(f"Updated album to: {album_number}")

def main():
    path_df = pd.read_csv('best_path.csv')
    path = 'podcast001/'

    for i in range(len(path_df)):
        file_name = path_df.iloc[i]['song']
        album_number = i + 1
        set_album(path + file_name, album_number)

if __name__ == "__main__":
    main()
