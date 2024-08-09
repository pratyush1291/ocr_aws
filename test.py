# import cv2
# from PyPDF2 import PdfWriter, PdfReader
# from multiprocessing import Pool,Manager,freeze_support
# from pdf2image import convert_from_path
# import easyocr
# import numpy as np
# from PIL import Image
# import os
# import glob
# import pandas as pd
# import re
# import datetime
# import cv2
# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename
# import shutil
#
#
#
#
# reader = easyocr.Reader(['en'])
# import os
#
# def count_file(directory):
#     if os.path.exists(directory):
#         return len(os.listdir(directory))
#     else:
#         return 0
# def count_files(directory):
#     files = os.listdir(directory)
#     num_files = len(files)
#     return num_files
# directory = "pdf"
# count = count_files(directory)
# count = int(count)
# # Read text from an image
# image_path = [f"house_number/unorder_output_image_number{j}/cropped_table_{i}.png"
#               for j in range(0, count)
#               if count_file(f'house_number/unorder_output_image_number{j}') > 0
#               for i in range(count_files(f'house_number/unorder_output_image_number{j}') - 1, -1, -1)]
#
# #reader2 = easyocr.Reader(['en'], recog_network='en_sample')
# reader1 = easyocr.Reader(['en'],recog_network='best_accuracy')
# data=[]
# directory=r'D:\ElvoteOCRModel\voter_number'
# #image_path=os.listdir(directory)
# print(image_path)
# for image in image_path:
#     #mage=os.path.join(directory,image)
#
#
#     # print(f"Reading Number")
#     # image = convert_to_supported_format(image)
#
#     # print(f"Reading text ")
#     result1 = reader1.readtext(image)
#     if result1:
#         #text_result2 = result1[0][1]
#         data.append(result1)
#     else:
#         print(result1)
#
#
# # Create DataFrame
# dataset = pd.DataFrame(data, columns=['Voter Number'])
# dataset.to_csv("voter_number.csv")

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

perc=fuzz.partial_ratio('CT', 'CTZ')
print(perc)