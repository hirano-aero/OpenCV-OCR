# -*- coding: utf-8 -*-

from PIL import Image
import sys

import pyocr
import pyocr.builders

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))

img = Image.open("wikipedia.jpg")
bul = pyocr.builders.TextBuilder(tesseract_layout=6)

txt = tool.image_to_string(
    img,
    lang="jpn",
    builder=bul
)
print( txt )