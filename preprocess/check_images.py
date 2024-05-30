import os
from PIL import Image, ImageFile
from tqdm import tqdm

def check_and_remove_images(directory):
    corrupted_count = 0
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # Do not load truncated images

    for subdir, dirs, files in os.walk(directory):
        for file in tqdm(files, position=0, leave=False):
            file_path = os.path.join(subdir, file)
            try:
                with Image.open(file_path) as img:
                    img.load()  # Attempt to load the image to catch truncation errors
            except (IOError, SyntaxError, ValueError) as e:
                print(f'Removing corrupted image: {file_path} with error {e}')
                os.remove(file_path)  # Remove the corrupted image
                corrupted_count += 1
    return corrupted_count


def main():
    parser = argparse.ArgumentParser(description="Check and remove corrupted images in a directory")
    parser.add_argument('directory', type=str, help='The path to the directory containing the images')
    
    args = parser.parse_args()
    
    corrupted_images_count = check_and_remove_images(args.directory)
    print(f'Number of corrupted images removed: {corrupted_images_count}')

if __name__ == '__main__':
    main()