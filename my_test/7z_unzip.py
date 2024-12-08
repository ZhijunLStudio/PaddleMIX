import py7zr

with py7zr.SevenZipFile('datasets/llava/valid_images.7z', mode='r') as z:
    z.extractall(path='datasets/llava')
