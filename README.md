# color_namer
From colors to names and from names to colors - applying deep learning to guess what color a "paint color" style name is ("Autumn Midnight" anyone?). Also takes a color and tries to generate a working name.

## Requirements
1. Download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
2. Unzip the Google News vectors, don't change the filename
3. Run "main" in color_namer.py alternatively:

## Colors -> Names
```python
# Import and Train Model
color_namer = ColorNamer()
color_namer.train_colors_to_words()

# Add names in a list of strings
color_namer.generate_color(["purple"], display=True)
```



## Names -> Colors
In progress

## Sources
*Heavily* based on this project: https://github.com/airalcorn2/Color-Names
