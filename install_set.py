import subprocess
import os

def main():
    # Create a directory for the podcast
    podcast_dir = 'podcast001'
    if not os.path.exists(podcast_dir):
        os.makedirs(podcast_dir)

    # Change the current working directory to the podcast directory
    os.chdir(podcast_dir)

    # List of SoundCloud URLs to download from
    urls = [
        "https://soundcloud.com/okky-070720/sets/podcast001",
        # "https://soundcloud.com/nicolas-ibanez-898141406/freeze-corleone-freeze-rael-techno-remix",
        # "https://soundcloud.com/nicolas-ibanez-898141406/preview-jour-de-plus-freeze-corleone"
    ]

    # Download each track/set from SoundCloud
    for url in urls:
        # subprocess.call(['scdl', '-l', url, '--download-archive', 'downloaded.txt'])
        print(f'Downloading "{url}"...')
        print(f"scdl -l {url}")
        subprocess.call(['scdl', '-l', url, '--download-archive', 'downloaded.txt'])

    playlist_name = 'Podcast001'

    os.chdir(playlist_name)

    # Rename files to remove playlist name prefix
    for filename in os.listdir('.'):
        if filename.startswith(playlist_name):
            new_name = filename.replace(playlist_name + '_', '')
            os.rename(filename, new_name)
            print(f'Renamed "{filename}" to "{new_name}"')
            # move it to the parent directory
            os.rename(new_name, '../' + new_name)
            print(f'Moved "{new_name}" to parent directory')

    # Delete the playlist directory
    os.chdir('..')
    os.rmdir(playlist_name)

if __name__ == '__main__':
    main()
