#%%
from PIL import Image
import os

def resize_images(directory, scale_factor):
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                img = img.resize((new_width, new_height))
                img.save(img_path)
                print(f"Resized {filename} to {new_width}x{new_height}")

#%%
graphs_directory = "d:/Repos/NIBS/graphs"
scale_factor = 0.75
resize_images(graphs_directory, scale_factor)
# %%
