import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from multiprocessing import Pool,Manager,freeze_support
import easyocr
import pytesseract
from pdf2image import convert_from_path
from pytesseract import image_to_string
import re
import concurrent.futures
import pandas as pd
import numpy as np
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process

import matplotlib.pyplot as plt
import shutil
from PIL import Image
import cv2
import os
from datetime import datetime
from PyPDF2 import PdfWriter, PdfReader
#import easyocr
import os
import glob

import re

# reader = easyocr.Reader(['en'], recog_network='en_sample')
# reader1 = easyocr.Reader(['en'])
#pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\Tesseract-OCR\tesseract.exe'

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def parsing(pdf_path):
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        output_filename = f'pdf/output_image_{i}.png'
        image.save(output_filename, 'png')






def count_files(directory):
    files = os.listdir(directory)
    num_files = len(files)
    return num_files

def process_image(img_path):
    try:
        image = cv2.imread(img_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            os.makedirs("images",exist_ok=True)
            output_dir = f'images/unorder_{os.path.basename(img_path)}'
            output_dir = output_dir.replace(".png", "")
            os.makedirs(output_dir, exist_ok=True)

            min_table_area = 1000
            j = 0
            for contour in contours:
                if cv2.contourArea(contour) > min_table_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_table = image[y:y + h, x:x + w]
                    output_path = os.path.join(output_dir, f'cropped_table_{j}.png')

                    j += 1
                    cv2.imwrite(output_path, cropped_table)
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            directory = "/home/ec2-user/pdf"
            items = os.listdir(directory)

            for k in range(len(items)):
                directory2 = f"/home/ec2-user/images/unorder_output_image_{k}"
                os.makedirs(directory2, exist_ok=True)
                for r in range(len(os.listdir(directory2))):
                    image_path = os.path.join(directory2, f'cropped_table_{r}.png')
                    image = cv2.imread(image_path)

                    # Specify the coordinates and dimensions of the ROI
                    x, y, width, height = 300, 10, 520, 28# Example coordinates and dimensions for voter number
                    x1, y1, width1, height1 = 85, 100, 300, 25  #for house   #for house

                    # Crop the ROI using array slicing

                    cropped_image = image[y:y + height, x:x + width]
                    cropped_image_house=image[y1:y1 + height1, x1:x1 + width1]
                    directory3 = f"home/ec2-user/voter_number/unorder_output_image_number{k}"
                    directory_house = f"home/ec2-user/house_number/unorder_output_image_number{k}"

                    os.makedirs(directory3, exist_ok=True)
                    os.makedirs(directory_house, exist_ok=True)

                    output_path2 = os.path.join(directory3, f'cropped_table_{r}.png')
                    output_path_house = os.path.join(directory_house, f'cropped_table_{r}.png')

                    # Save the cropped image
                    cv2.imwrite(output_path2, cropped_image)
                    cv2.imwrite(output_path_house,cropped_image_house)
    except Exception as e:
        print(e)








def process_image_english(img_path):
    try:
        image = cv2.imread(img_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            os.makedirs("images",exist_ok=True)
            output_dir = f'images/unorder_{os.path.basename(img_path)}'
            output_dir = output_dir.replace(".png", "")
            os.makedirs(output_dir, exist_ok=True)

            min_table_area = 1000
            j = 0
            for contour in contours:
                if cv2.contourArea(contour) > min_table_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped_table = image[y:y + h, x:x + w]
                    output_path = os.path.join(output_dir, f'cropped_table_{j}.png')

                    j += 1
                    cv2.imwrite(output_path, cropped_table)
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

            directory = "home/ec2-user/pdf"
            items = os.listdir(directory)

            for k in range(len(items)):
                directory2 = f"home/ec2-user/images/unorder_output_image_{k}"
                os.makedirs(directory2, exist_ok=True)
                for r in range(len(os.listdir(directory2))):
                    image_path = os.path.join(directory2, f'cropped_table_{r}.png')
                    image = cv2.imread(image_path)

                    # Specify the coordinates and dimensions of the ROI
                    x, y, width, height = 300, 10, 520, 28# Example coordinates and dimensions for voter number


                    # Crop the ROI using array slicing

                    cropped_image = image[y:y + height, x:x + width]

                    directory3 = f"home/ec2-user/voter_number/unorder_output_image_number{k}"


                    os.makedirs(directory3, exist_ok=True)


                    output_path2 = os.path.join(directory3, f'cropped_table_{r}.png')


                    # Save the cropped image
                    cv2.imwrite(output_path2, cropped_image)

    except Exception as e:
        print(e)
















def extract_data(img_path):
        #reader = easyocr.Reader(['en'])

        words_to_remove = ["Photo", "Available", "Availabl"]
        # Name = []
        # RelationName = []
        HouseNumber = []
        # Age = []
        # Gender = []






        # pattern_name = r'Name :(.*)'
        # pattern_fathername = r'Fathers Name:(.*)'
        # pattern_fathername2 = r"Father's Name:(.*)"
        # pattern_husband = r'Husbands Name:(.*)'
        # pattern_husband2 = r"Husband's Name:(.*)"
        #
        # pattern_Others = r'Others:(.*)'
        # pattern_Others2 = r"Other's:(.*)"
        # pattern_Mother = r'Mothers Name:(.*)'
        # pattern_Mother2 = r"Mother's Name:(.*)"
        pattern_house_number = r'House Number :(.*)'
        # pattern_age = r'Age :(.{3})'
        # pattern_gender = r'Gender :(.*)'

        pattern = "|".join(words_to_remove)
        pattern2 = r"\[[^\]]*\]"
        pattern3 = r"\([^)]*\]"
        # pattern_type1 = r'[A-Z]+\d+'
        # pattern_type2 = r'[A-Z]{2}/\d{2}/\d{3}/\d+'








        image_pil = Image.open(img_path)
        #
        if image_pil is not None:
           width, height = image_pil.size
           width_70_percent = int(width * 0.70)
           width_40_percent = int(width * 0.30)
           left_portion = image_pil.crop((0, 0, width_70_percent, height))

        #     right_portion = image_pil.crop((width - width_40_percent, 0, width,new_height))
            #image = cv2.cvtColor(np.array(right_portion), cv2.COLOR_RGB2BGR)
        # #target_size = (800, 400)
       #image = cv2.resize(image, target_size)
            # gray_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2GRAY)
            # blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            # alpha = 1.1  # Controls contrast (1.0 is the original contrast)
            # beta = 0  # Controls brightness (0 is no brightness, 100 is full brightness)
            # enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=alpha, beta=beta)
            # binary_image = cv2.adaptiveThreshold(
            #      enhanced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)


        text = pytesseract.image_to_string(img_path, lang='eng')
        text = re.sub(pattern, "", text)
        text = re.sub(pattern2, "", text)
        text = re.sub(pattern3, "", text)
        text = text.replace("+", ":")
        text = text.replace("!", ":")
        text = text.replace("}", ":")
        text = text.replace("=", ":")
        text = text.replace("?", ":")
        text = text.replace("*", ":")
        text = re.sub(pattern, "", text)

        # name_match = re.search(pattern_name, text)
        # if name_match:
        #     Name.append(name_match.group(1).strip())
        #
        #
        # else:
        #     Name.append("M")
        #
        # fathername_match = re.search(pattern_fathername, text)
        # fathername2_match = re.search(pattern_fathername2, text)
        # husband_match = re.search(pattern_husband, text)
        # husband2_match = re.search(pattern_husband2, text)
        # others_match = re.search(pattern_Others, text)
        # others2_match = re.search(pattern_Others2, text)
        # mother_match = re.search(pattern_Mother, text)
        # mother2_match = re.search(pattern_Mother2, text)
        #
        # if fathername_match:
        #     RelationName.append(fathername_match.group(1).strip())
        # elif husband_match:
        #     RelationName.append(husband_match.group(1).strip())
        # elif others_match:
        #     RelationName.append(others_match.group(1).strip())
        # elif mother_match:
        #     RelationName.append(mother_match.group(1).strip())
        # elif fathername2_match:
        #     RelationName.append(fathername2_match.group(1).strip())
        # elif husband2_match:
        #     RelationName.append(husband2_match.group(1).strip())
        # elif others2_match:
        #     RelationName.append(others2_match.group(1).strip())
        # elif mother2_match:
        #     RelationName.append(mother2_match.group(1).strip())
        # else:
        #     RelationName.append("M")

        text2 = pytesseract.image_to_string(left_portion, lang='eng')

        house_number_match = re.search(pattern_house_number, text2)
        if house_number_match:
            HouseNumber.append(str(house_number_match.group(1).strip()))
        else:
            HouseNumber.append("M")

        # age_match = re.search(pattern_age, text)
        # if age_match:
        #     Age.append(age_match.group(1).strip())
        # else:
        #     Age.append("M")
        #
        # gender_match = re.search(pattern_gender, text)
        # if gender_match:
        #     Gender.append(gender_match.group(1).strip())
        # else:
        #     Gender.append("M")
        #
        # max_length = max(len(Name), len(RelationName), len(HouseNumber), len(Age), len(Gender))
        # Name += ["M"] * (max_length - len(Name))
        # RelationName += ["M"] * (max_length - len(RelationName))
        # HouseNumber += ["M"] * (max_length - len(HouseNumber))
        # Age += ["M"] * (max_length - len(Age))
        # Gender += ["M"] * (max_length - len(Gender))


        data1 = {


                #"Name": Name,
                #'RelationName': RelationName,
                "House Number": HouseNumber,
                #"Age": Age,
                #"Gender": Gender
        }
        return data1
d1=[]
d2=[]
d3=[]
d4=[]


# def extract_data_voter_number(image):
#     serial_numbers = []
#
#     result = reader.readtext(image)
#     # print(f"Reading text ")
#     result1 = reader1.readtext(image)
#
#     if result and result1:
#         # Extract and match the patterns
#         text_result = result[0][1]
#         text_result1 = result1[0][1]
#
#         match_number = re.search(r'([0-9]{7})', text_result)
#         match_letters = re.match(r'([a-zA-Z]{3})', text_result1)
#         # print(f"Match Letters: {text_result1} Match Letters: {match_letters}")
#         if match_number and match_letters:
#             voter_number = match_letters.group(1).upper() + match_number.group(1)
#             print(f"image pathe:{image}  Voter Number: {voter_number}")
#             serial_numbers.append(voter_number)
#
#     data3={
#         'serial_numbers':serial_numbers
#     }
#
#     return data3







def pooling_stations(args):
    try:
        imgpath,imgpath2 = args
        d3 = []


        counter = 0



        pattern_assembly_station = r'AssemblyConstituencyNoandName:(.*)'
        pattern_part = r'PartNo.:(.*)'
        section_part = r'SectionNoandName(.*)'
        #section_part2=r'SectionNoandName3(.*)'
        section_part2=r'1-ListofAdditions2(.*)'
        text = pytesseract.image_to_string(imgpath, lang='eng')
        textdata = " ".join(text)
        textdata = textdata.replace(" ", "")

        # match = re.search(pattern_assembly_station, textdata)
        # data2 = match.group(1).strip()
        #
        # match2 = re.search(pattern_part, data2)
        # if match2 == None:
        #     assembly = data2
        #     match3 = re.search(pattern_part, textdata)
        #     if match3 == None:
        #         missing = "New"
        #         Part_No = missing
        #     else:
        #         Part_No = match3.group(1).strip()
        #
        # else:
        #
        #     number = match2.group(1).strip()
        #     # print(number)
        #     Part_No = number
        #     data2 = data2.replace("PartNo.:" + number, "")
        #     assembly = data2

        match4 = re.search(section_part, textdata)
        matchnew=re.search(section_part2,textdata)
        if match4 == None and matchnew ==None:
            missing = "New"
            section = missing

        elif match4:
                section = match4.group(1).strip()

        else:
            section = matchnew.group(1).strip()


        new_dir="home/ec2-user/pdf"
        count = count_files(new_dir)
        count = int(count)
        directory =imgpath2
        count = count_files(directory)
        count = int(count)
        for k in range(count, 0, -1):





                d3.append(section)



    except Exception as e:
        print(e)

    return  d3

def extract_data2(img_path):

        words_to_remove = ["Photo", "Available", "Availabl"]
        Name = []
        RelationName = []
        HouseNumber = []
        Age = []
        Gender = []
        serial_numbers = []

        pattern_name = r'Name :(.*)'
        pattern_fathername = r'Fathers Name:(.*)'
        pattern_fathername2 = r"Father's Name:(.*)"
        pattern_husband = r'Husbands Name:(.*)'
        pattern_husband2 = r"Husband's Name:(.*)"

        pattern_Others = r'Others:(.*)'
        pattern_Others2 = r"Other's:(.*)"
        pattern_Mother = r'Mothers Name:(.*)'
        pattern_Mother2 = r"Mother's Name:(.*)"
        pattern_house_number = r'House Number :(.*)'
        pattern_age = r'Age :(.{3})'
        pattern_gender = r'Gender :(.*)'

        pattern = "|".join(words_to_remove)
        pattern2 = r"\[[^\]]*\]"
        pattern3 = r"\([^)]*\]"
        pattern_type1 = r'[A-Z]+\d+'


        image_pil = Image.open(img_path)
        width, height = image_pil.size
        width_70_percent = int(width * 0.70)
        width_40_percent = int(width * 0.30)
        left_portion = image_pil.crop((0, 0, width_70_percent, height))
        # new_height = int(height * 0.2)
        # #right_portion = image_pil.crop((width - width_40_percent, 0, width,new_height))
        # #
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        target_size = (800, 400)
        image = cv2.resize(image, target_size)
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        alpha = 1.1  # Controls contrast (1.0 is the original contrast)
        beta = 40  # Controls brightness (0 is no brightness, 100 is full brightness)
        enhanced_image = cv2.convertScaleAbs(blurred_image, alpha=alpha, beta=beta)
        binary_image = cv2.adaptiveThreshold(
            enhanced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)





        text = pytesseract.image_to_string(left_portion, lang='eng')
        text = re.sub(pattern, "", text)
        text = re.sub(pattern2, "", text)
        text = re.sub(pattern3, "", text)
        text = text.replace("+", ":")
        text = text.replace("!", ":")
        text = text.replace("}", ":")
        text = text.replace("=", ":")
        text = text.replace("?", ":")
        text = text.replace("*", ":")
        text = re.sub(pattern, "", text)


        house_number_match = re.search(pattern_house_number, text)
        if house_number_match:
             HouseNumber.append(str(house_number_match.group(1).strip()))
        else:
             HouseNumber.append("M")


        data2 = {


             "House Number": HouseNumber,

        }

        return data2





#hindi ocr

#reader1 = easyocr.Reader(['en'],gpu=False)
#reader2 = easyocr.Reader(['en'], recog_network='en_sample',gpu=False)
#reader4=easyocr.Reader(['en'],recog_network='en_sample',gpu=False)


def easy_ocr_hindi(image):

    data=[]

    result = reader2.readtext(image)


    # result1 = reader1.readtext(image)
    # print(result1)

    # if result and result1:
    #     # Extract and match the patterns
    #     text_result = result[0][1]
    #     text_result1 = result1[0][1]
    #
    #
    #     if len(text_result1) >12 or len(text_result) >12:
    #         result2=reader4.readtext(image)
    #         text_result2=result2[0][1]
    #         voter_number=text_result2
    #
    #         data.append(voter_number)
    #     else:
    #
    #         match_number = re.search(r'([0-9]{7})', text_result)
    #         match_letters = re.match(r'([a-zA-Z]{3})', text_result1)
    #         #print(f"Match Letters: {text_result1} Match Letters: {match_letters}")
    #         if match_number and match_letters:
    #             voter_number = match_letters.group(1).upper() + match_number.group(1)
    #
    #         else:
    #             voter_number=text_result1
    text_result = result[0][1]

    data.append(text_result)


    return data


def housenumber_hindi(image):
    try:

        house_data=[]
        text = pytesseract.image_to_string(image, lang='Devanagari')
        text2= pytesseract.image_to_string(image, lang='hin')

        if text:

            house_data.append(text)

        elif text2:
            house_data.append(text2)
        else:
            result = reader1.readtext(image)
            text_result = result[0][1]
            house_data.append(text_result)

    except Exception as e:
        print(e)



    return house_data


def pooling_stations_hindi(args):
    imgpath,imgpath2 = args

    d3 = []
    d4 = []
    counter = 0




    #pattern_assembly_station = r'विधानसभा निर्वाचन क्षेत्र की संख्या, नाम :(.*)'
    #pattern_part = r'भाग संख्या :(.*)'
    section_part = r'अनुभागों की संख्या व नाम(.*)'
    text = pytesseract.image_to_string(imgpath, lang='hin')



    #match = re.search(pattern_assembly_station, textdata)
    #print(match)


    #data2 = match.group(1).strip()

    #match2 = re.search(pattern_part, data2)
    # if match2 == None:
    #     assembly = data2
    #     match3 = re.search(pattern_part, textdata)
    #     if match3 == None:
    #         missing = "New"
    #         Part_No = missing
    #     else:
    #         Part_No = match3.group(1).strip()
    #
    # else:
    #
    #     number = match2.group(1).strip()
    #     # print(number)
    #     Part_No = number
    #     data2 = data2.replace("PartNo.:" + number, "")
    #     assembly = data2

    match4 = re.search(section_part, text)


    if match4 == None:
        missing = "New"
        section = missing
    else:
        section = match4.group(1).strip()
    new_dir="home/ec2-user/pdf"
    count = count_files(new_dir)
    count = int(count)
    directory =imgpath2
    count = count_files(directory)
    count = int(count)
    for k in range(count, 0, -1):
            counter = counter + 1
            #d1.append(assembly)

            #d2.append(Part_No)

            d3.append(section)
            d4.append(counter)
    # data = {
    #     "Number": d4,
    #     "assembly": d1,
    #     "Part_No": d2,
    #     "section": d3,
    # }
    return d3,d4

def extract_data2_hindi(args):
        image,img_path=args


        words_to_remove = ["फोटो उपलब्ध", "फोटो उपसब्ध", "फोटो उपलय्ध"]

        HouseNumber = []



        pattern_house_number = r'मकान संख्या :(.*)'
        pattern_house_number_2 = r'मकान संख्या (.*)'
        pattern_house_number_5 = r'मकान नं.(.*)'
        pattern_house_number_3 = r'मकान नं.:(.*)'
        pattern_house_number_4 = r'मकान नं.:(.*)'

        pattern = "|".join(words_to_remove)
        pattern2 = r"\[[^\]]*\]"
        pattern3 = r"\([^)]*\]"
        pattern_type1 = r'[A-Z]+\d+'

        image_pil = Image.open(img_path)
        width, height = image_pil.size
        width_70_percent = int(width * 0.70)
        width_40_percent = int(width * 0.30)
        left_portion = image_pil.crop((0, 0, width_70_percent, height))





        text = pytesseract.image_to_string(left_portion, lang='Devanagari')

        text = re.sub(pattern, "", text)
        text = re.sub(pattern2, "", text)
        text = re.sub(pattern3, "", text)
        text = text.replace("+", ":")
        text = text.replace("!", ":")
        text = text.replace("}", ":")
        text = text.replace("=", ":")
        text = text.replace("?", ":")
        text = text.replace("*", ":")
        text = re.sub(pattern, "", text)


        house_number_match = re.search(pattern_house_number, text)
        if house_number_match:
            HouseNumber.append(str(house_number_match.group(1).strip()))
        else:
            house_number_match = re.search(pattern_house_number_2, text)
            if house_number_match:
                HouseNumber.append(str(house_number_match.group(1).strip()))
            else:
                house_number_match = re.search(pattern_house_number_3, text)
                if house_number_match:
                    HouseNumber.append(str(house_number_match.group(1).strip()))
                else:
                    house_number_match = re.search(pattern_house_number_4, text)
                    if house_number_match:
                        HouseNumber.append(str(house_number_match.group(1).strip()))
                    else:
                        house_number_match = re.search(pattern_house_number_5, text)
                        if house_number_match:
                            HouseNumber.append(str(house_number_match.group(1).strip()))
                        else:
                            text = pytesseract.image_to_string(image, lang='Devanagari')
                            text2 = pytesseract.image_to_string(image, lang='hin')

                            if text:

                                HouseNumber.append(text)

                            elif text2:
                               HouseNumber.append(text2)
                            else:

                                    text_result="M"
                                    HouseNumber.append(text_result)





        return HouseNumber











@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part"}), 400

    language = request.form.get('language')
    print(f'Selected language: {language}')

    files = request.files.getlist('files[]')

    file_paths = []

    if language=="english":
        start=datetime.now()
        dir = "home/ec2-user/data"
        os.makedirs(dir,exist_ok=True)
        print("i am here")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)
        data_frames = []
        data_frames2=[]
        data_frame_voter = []

        directory_d = 'home/ec2-user/data'
        combined_df = pd.DataFrame()

        total_pdfs = []
        for filename in os.listdir(directory_d):
            filename = "data/" + filename
            total_pdfs.append(filename)

        for pdf in total_pdfs:
            pdf_path = pdf
            images = convert_from_path(pdf_path, first_page=1, last_page=3)

            # Initialize an empty string to store extracted text
            extracted_text = ''
            count = 0
            # Loop through each page/image
            section_part = r'Section No and Name(.*)'
            addition_part = r'List of Additions 1'
            addition_part2=r'List of Additions 2'
            for img in images:
                # Extract text from image using Pytesseract
                text = pytesseract.image_to_string(img, lang='eng')  # You can specify the language
                extracted_text += text

            match4 = re.search(section_part, extracted_text)

            # Print or save the extracted text

            output_pdf_path = pdf_path
            # when it is 3 page to delete match4 will return empty
            if match4:

                extracted_text3 = ''
                extracted_text2=''

                with open(pdf_path, 'rb') as infile:
                    reader = PdfReader(infile)
                    writer = PdfWriter()
                    lastpage = len(reader.pages)
                    firstpage = lastpage - 2     #total last 3 pages will be checked

                    images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                    for img in images:
                        text = pytesseract.image_to_string(img, lang='eng')  # You can specify the language
                        extracted_text3 =extracted_text3+ text

                    match5 = re.search(section_part, extracted_text3)
                    match6 = re.search(addition_part, extracted_text3)
                    match7 = re.search(addition_part2,extracted_text3)
                    if match5 or match6 or match7:
                        #where we have to delete two or one page
                        lastpage=len(reader.pages)
                        firstpage-lastpage-1
                        images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                        for img in images:
                            text = pytesseract.image_to_string(img, lang='eng')  # You can specify the language
                            extracted_text2 = extracted_text2 + text
                        match5 = re.search(section_part, extracted_text2)
                        match6 = re.search(addition_part, extracted_text2)
                        match7 = re.search(addition_part2, extracted_text2)

                        if match5 or match6 or match7:

                            for page_num in range(2,lastpage-2):
                                writer.add_page(reader.pages[page_num])
                            with open(output_pdf_path, 'wb') as outfile:
                                writer.write(outfile)
                        else:
                            for page_num in range(2, lastpage-1):
                                writer.add_page(reader.pages[page_num])
                            with open(output_pdf_path, 'wb') as outfile:
                                    writer.write(outfile)
                    else:
                        #delete last 3 pages

                        for page_num in range(2, lastpage - 3):
                            writer.add_page(reader.pages[page_num])

                        with open(output_pdf_path, 'wb') as outfile:
                            writer.write(outfile)

            # If match4 is not found, inform the user
            elif not match4:

                extracted_text3 = ''
                extracted_text2=''
                with open(pdf_path, 'rb') as infile:
                    reader = PdfReader(infile)
                    writer = PdfWriter()
                    lastpage = len(reader.pages)
                    firstpage = lastpage - 2
                    images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                    for img in images:
                        text = pytesseract.image_to_string(img, lang='eng')  # You can specify the language
                        extracted_text3=extracted_text3+text
                    match5 = re.search(section_part, extracted_text3)
                    match6 = re.search(addition_part, extracted_text3)
                    match7 = re.search(addition_part2, extracted_text3)
                    if match5 or match6 or match7:
                        lastpage=len(reader.pages)
                        firstpage=lastpage-1   #check last 3 pages
                        images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                        for img in images:
                            text = pytesseract.image_to_string(img, lang='eng')  # You can specify the language
                            extracted_text2 = extracted_text2 + text
                        match5 = re.search(section_part, extracted_text2)
                        match6 = re.search(addition_part, extracted_text2)
                        match7 = re.search(addition_part2, extracted_text2)

                        if match5 or match6 or match7:
                            for page_num in range(3, lastpage-1):
                                writer.add_page(reader.pages[page_num])

                            with open(output_pdf_path, 'wb') as outfile:
                                writer.write(outfile)
                        else:

                            for page_num in range(3, lastpage - 2):
                                writer.add_page(reader.pages[page_num])

                            with open(output_pdf_path, 'wb') as outfile:
                                writer.write(outfile)
                    else:


                        #will delete last 3 pages

                        for page_num in range(3, lastpage - 3):
                            writer.add_page(reader.pages[page_num])

                        with open(output_pdf_path, 'wb') as outfile:
                            writer.write(outfile)


            counter = 0
            pdf_path = pdf
            directory = "pdf"
            if os.path.exists(directory):
                # Remove the existing directory and its contents
                shutil.rmtree(directory)
            os.makedirs('pdf', exist_ok=True)

            parsing(pdf_path)
            # directory = "pdf"
            count = count_files(directory)
            count = int(count)

            # Create a Pool of worker processes


            # Process images in parallel

            try:
                pool = Pool()
                image_paths = [f"pdf/output_image_{i}.png" for i in range(0, count)]
                pool.map(process_image_english, image_paths)
                pool.close()
            except Exception as e:
                print(f"An error occurred during multiprocessing: {e}. Continuing with the rest of the images.")



            #Mutti-Threading
            # image_paths = [f"pdf/output_image_{i}.png" for i in range(0, count)]
            #
            # with concurrent.futures.ThreadPoolExecutor(max_workers=250) as executor:
            #     executor.map(process_image, image_paths)



            # Extract data from processed images in parallel
            def count_file(directory):
                if os.path.exists(directory):
                    return len(os.listdir(directory))
                else:
                    return 0
            try:
                pool10=Pool()


                image_path = [f"images/unorder_output_image_{j}/cropped_table_{i}.png"
                              for j in range(0, count)
                              if count_file(f'images/unorder_output_image_{j}') > 0
                              for i in range(count_files(f'images/unorder_output_image_{j}') - 1, -1, -1)]

                house = pool10.map(extract_data, image_path)
                house = pd.DataFrame(house)
                data_frames.append(house)
                pool10.close()
            except Exception as e:
                print(e)




            # try:
            #
            #     pool3 = Pool()
            #
            image_path = [f"voter_number/unorder_output_image_number{j}/cropped_table_{i}.png"
                              for j in range(0, count)
                              if count_file(f'voter_number/unorder_output_image_number{j}') > 0
                              for i in range(count_files(f'voter_number/unorder_output_image_number{j}') - 1, -1, -1)]

            reader1 = easyocr.Reader(['en'], gpu=False)
            reader2 = easyocr.Reader(['en'], recog_network='en_sample', gpu=False)
            reader4 = easyocr.Reader(['en'], recog_network='en_sample', gpu=False)
            data = []
            for image in image_path:
                # print(f"Reading Number")
                result = reader2.readtext(image)
                # print(f"Reading text ")
                result1 = reader1.readtext(image)

                if result and result1:
                    # Extract and match the patterns
                    text_result = result[0][1]
                    text_result1 = result1[0][1]

                    if len(text_result1) > 12 or len(text_result) > 12:
                        result2 = reader4.readtext(image)
                        text_result2 = result2[0][1]
                        voter_number = text_result2

                        data.append(voter_number)
                    else:

                        match_number = re.search(r'([0-9]{7})', text_result)
                        match_letters = re.match(r'([a-zA-Z]{3})', text_result1)
                        # print(f"Match Letters: {text_result1} Match Letters: {match_letters}")
                        if match_number and match_letters:
                            voter_number = match_letters.group(1).upper() + match_number.group(1)

                        else:
                            voter_number = text_result1

                        data.append(voter_number)
                        # print(f"image pathe:{image}  Voter Number: {voter_number}")



            # Create DataFrame
            voter = pd.DataFrame(data)
            data_frame_voter.append(voter)


            # image_path_house = [f"house_number/unorder_output_image_number{j}/cropped_table_{i}.png"
            #               for j in range(0, count)
            #               if count_file(f'house_number/unorder_output_image_number{j}') > 0
            #               for i in range(count_files(f'house_number/unorder_output_image_number{j}') - 1, -1, -1)]
            #
            #
            # data2=[]
            # for image in image_path_house:
            #     # print(f"Reading Number")
            #
            #     # print(f"Reading text ")
            #     result1 = reader1.readtext(image)
            #
            #     data2.append(result1)
            #
            #
            # # Create DataFrame
            # dataset = pd.DataFrame(data2, columns=['house Number'])
            # dataset.to_csv('house_number.csv', index=False)
            #


            #
            #     data3 = pool3.map(easy_ocr_hindi, image_path)
            #
            #     pool3.close()
            # except Exception as e:
            #     print(e)
            #
            #
            #
            # df = pd.DataFrame(data1)
            # df_voter=pd.DataFrame(data3)
            #
            #
            # data_frames.append(df)
            # data_frame_voter.append(df_voter)
            # #
            #
            pool2 = Pool()
            try:

                imgpath = [f"home/ec2-user/pdf/output_image_{i}.png" for i in range(count)]

                imgpath2 = [f"home/ec2-user/images/unorder_output_image_{i}" for i in range(count)]

                args_list = list(zip(imgpath, imgpath2))

                results = pool2.map(pooling_stations, args_list)

                # df = pd.DataFrame(data)
                # df=df.transpose()
                # print(df)



                pooling_station = []

                for result in results:


                    pooling_station.extend(result)
                df_combined = pd.DataFrame({'section': pooling_station})
                # print(df_combined)

                # df_column = df1.transpose()
                data_frames2.append(df_combined)


                # Create a DataFrame from the collected data

                #df_combined.to_excel('combined_station.xlsx', index=False)

                pool2.close()
                pool2.join()
            except Exception as e:
                print(e)
            #
            # # try:
            # #
            # #     pool3 = Pool()
            # #
            # #
            # #     image_path = [f"voter_number/unorder_output_image_number{j}/cropped_table_{i}.png"
            # #                   for j in range(0, count)
            # #                   if count_file(f'voter_number/unorder_output_image_number{j}') > 0
            # #                   for i in range(count_files(f'voter_number/unorder_output_image_number{j}') - 1, -1, -1)]
            # #
            # #     data3 = pool3.map(easy_ocr_hindi, image_path)
            # #
            # #
            # #
            # #     pool3.close()
            # # except Exception as e:
            # #     print(e)




            shutil.rmtree('home/ec2-user/images')
            shutil.rmtree('home/ec2-user/voter_number')
            #shutil.rmtree('D:/ocr_production_hosting/house_number')
            dir2 = "home/ec2-user/pdf"
            shutil.rmtree(dir2)
            end=datetime.now()
            print("total time for image processing",end-start)





            # print("Total time spent:", total)
        final_df = pd.concat(data_frames, ignore_index=True)
        final_df.to_excel('combined_voter.xlsx', index=False)
        final_df2=pd.concat(data_frames2,ignore_index=True)
        final_df2.to_excel('combined_polling_station.xlsx', index=False)
        final_df3 = pd.concat(data_frame_voter, ignore_index=True)
        final_df3.to_excel('epic_number.xlsx', index=False)

        shutil.rmtree('home/ec2-user/data')
        dir = "home/ec2-user/data"
        os.makedirs(dir, exist_ok=True)


        # dir2 = "D:/ocr_production_hosting/pdf"
        # shutil.rmtree(dir2)
        #
        # shutil.rmtree('D:/ocr_production_hosting/images')
        # shutil.rmtree('D:/ocr_production_hosting/voter_number')

        return jsonify({"message": "Files successfully uploaded with ocr", "file_paths": file_paths})

    elif language=="hindi":
        start = datetime.now()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)
        data_frames = []
        data_frames2=[]
        data_frame_voter=[]


        pattern=[]
        counter=0

        directory_d = 'home/ec2-user/data'
        combined_df = pd.DataFrame()

        total_pdfs = []
        for filename in os.listdir(directory_d):
            filename = "data/" + filename
            total_pdfs.append(filename)

        for pdf in total_pdfs:
            counter=counter+1

            pdf_path = pdf
            images = convert_from_path(pdf_path, first_page=1, last_page=3)

            # Initialize an empty string to store extracted text
            extracted_text = ''

            # Loop through each page/image
            section_part = r'अनुभागों की संख्या व नाम'
            addition_part = r'जोड़े गए नामों की सूची 1'
            addition_part2=r'जोड़े गए नामों की सूची 2'

            for img in images:
                # Extract text from image using Pytesseract
                text = pytesseract.image_to_string(img, lang='hin')  # You can specify the language
                extracted_text += text


            match4 = re.search(section_part, extracted_text)

            # Print or save the extracted text

            output_pdf_path = pdf_path
            # when it is 3 page to delete match4 will return empty

            if match4:
                # when we have to delete the first two pages it comes here


                extracted_text3 = ''
                extracted_text2 = ''

                with open(pdf_path, 'rb') as infile:
                    reader = PdfReader(infile)
                    writer = PdfWriter()
                    lastpage = len(reader.pages)
                    firstpage = lastpage - 2   # total last 3 pages will be checked

                    images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                    for img in images:
                        text = pytesseract.image_to_string(img, lang='hin')  # You can specify the language
                        extracted_text3 =extracted_text3+text

                    match5 = re.search(section_part, extracted_text3)
                    match6 = re.search(addition_part, extracted_text3)
                    match7 = re.search(addition_part2, extracted_text3)
                    if match5 or match6 or match7:
                        #where we have to delete two or one page in the last page

                        lastpage = len(reader.pages)
                        firstpage = lastpage - 1
                        images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                        for img in images:
                            text = pytesseract.image_to_string(img, lang='hin')  # You can specify the language
                            extracted_text2 += text
                        match5 = re.search(section_part, extracted_text2)
                        match6 = re.search(addition_part, extracted_text2)
                        match7 = re.search(addition_part2, extracted_text2)
                        if match5 or match6 or match7:
                            print("i am here you are right ")
                            for page_num in range(2, lastpage-1):
                                writer.add_page(reader.pages[page_num])
                            with open(output_pdf_path, 'wb') as outfile:
                                    writer.write(outfile)
                        else:
                                for page_num in range(2, lastpage - 2):
                                    writer.add_page(reader.pages[page_num])
                                with open(output_pdf_path, 'wb') as outfile:
                                        writer.write(outfile)
                    else:



                        for page_num in range(2, lastpage-3):
                            writer.add_page(reader.pages[page_num])

                        with open(output_pdf_path, 'wb') as outfile:
                            writer.write(outfile)

            # If match4 is not found, inform the user
            elif not match4:

                extracted_text3 = ''
                extracted_text2=''

                with open(pdf_path, 'rb') as infile:
                    reader = PdfReader(infile)
                    writer = PdfWriter()
                    lastpage = len(reader.pages)
                    firstpage = lastpage - 2
                    print(firstpage)
                    images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)
                    for img in images:
                        text = pytesseract.image_to_string(img, lang='hin')  # You can specify the language
                        extracted_text3=extracted_text3+text
                    match5 = re.search(section_part, extracted_text3)
                    match6 = re.search(addition_part, extracted_text3)
                    match7 = re.search(addition_part2, extracted_text3)
                    if match5 or match6 or match7:
                        lastpage = len(reader.pages)
                        firstpage = lastpage - 1
                        images = convert_from_path(pdf_path, first_page=firstpage, last_page=lastpage)

                        # where we have to delete two or one page in the last page
                        for img in images:
                            text = pytesseract.image_to_string(img, lang='hin')  # You can specify the language
                            extracted_text2 += text
                        match5 = re.search(section_part, extracted_text2)
                        match6 = re.search(addition_part, extracted_text2)
                        match7 = re.search(addition_part2, extracted_text2)
                        if match5 or match6 or match7:
                            for page_num in range(3, lastpage-1):
                                writer.add_page(reader.pages[page_num])
                            with open(output_pdf_path, 'wb') as outfile:
                                    writer.write(outfile)
                        else:
                            for page_num in range(3, lastpage - 2):
                                    writer.add_page(reader.pages[page_num])
                            with open(output_pdf_path, 'wb') as outfile:
                                        writer.write(outfile)






                    else:
                        print("delete last 3 pages")

                        for page_num in range(3, len(reader.pages) - 3):
                            writer.add_page(reader.pages[page_num])

                        with open(output_pdf_path, 'wb') as outfile:
                            writer.write(outfile)

            counter = 0
            pdf_path = pdf
            directory = "pdf"
            if os.path.exists(directory):
                # Remove the existing directory and its contents
                shutil.rmtree(directory)
            os.makedirs('pdf', exist_ok=True)

            parsing(pdf_path)
            # directory = "pdf"
            count = count_files(directory)
            count = int(count)

            # Create a Pool of worker processes


            # Process images in parallel
            pool = Pool()
            try:

                image_paths = [f"pdf/output_image_{i}.png" for i in range(0, count)]
                pool.map(process_image, image_paths)
                pool.close()
                pool.join()
            except Exception as e:
                print(e)


            # Extract data from processed images in parallel
            def count_file(directory):
                if os.path.exists(directory):
                    return len(os.listdir(directory))
                else:
                    return 0
            # image_path = [f"voter_number/unorder_output_image_number{j}/cropped_table_{i}.png"
            #               for j in range(0, count)
            #               if count_file(f'voter_number/unorder_output_image_number{j}') > 0
            #               for i in range(count_files(f'voter_number/unorder_output_image_number{j}') - 1, -1, -1)]
            #
            reader1 = easyocr.Reader(['hi'], gpu=False)
            # reader2 = easyocr.Reader(['en'], recog_network='en_sample', gpu=True)
            # reader4 = easyocr.Reader(['en'], recog_network='best_accuracy', gpu=True)
            data = []
            # data1=[]
            #
            # try:
            #
            #     for image in image_path:
            #         img = cv2.imread(image)
            #         height, width, _ = img.shape
            #
            #         # Define cropping coordinates based on image width
            #         coordinates = [0, 2, 113, 58] if width <= 201 else [0, 2, 152,58]
            #
            #         # Crop the image
            #         cropped_img = img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
            #         # print(f"Reading Number")
            #         result = reader2.readtext(image)
            #
            #         # print(f"Reading text ")
            #
            #
            #
            #
            #             # Extract and match the patterns
            #         if  result:
            #             text_result = result[0][1]
            #             #print(text_result[-7:])
            #             #text_result1 = result1[0][1]
            #             if len(text_result) > 12 or len(text_result) > 12:
            #                 result2 = reader2.readtext(image)
            #                 text_result2 = result2[0][1]
            #                 voter_number = text_result2
            #
            #                 data.append(voter_number)
            #
            #             else:
            #                 result1 = reader4.readtext(cropped_img)
            #                 if result1:
            #                     text_result1 = result1[0][1]
            #                 else:
            #
            #
            #                     # Display the image using Matplotlib
            #
            #                     text_result1="M"
            #
            #
            #                 voter_number = text_result1[0:3]+text_result[-7:]
            #
            #
            #                 data.append(voter_number)
            # except Exception as e:
            #     print(e)

                # elif result:
                #     # result = reader1.readtext(image)
                #     #
                #     # text_result = result[0][1]
                #     #
                #     # voter_number=text_result.upper()
                #     # data.append(voter_number)
                #     # data1.append(voter_number2)
                #     # result5 = reader2.readtext(image)
                #
                #     text_result2 = result[0][1]
                #     voter_number = text_result2
                #
                #     data.append(voter_number)
                #     data1.append(voter_number)
                # else:
                #     result = reader1.readtext(image)
                #
                #     text_result = result[0][1]
                #
                #     voter_number=text_result.upper()
                #     print(voter_number)
                #     data.append(voter_number)
                #     data1.append(voter_number)








                    #print(text_result1)



            # try:
            #     pool2=Pool()
            #     data=pool2.map(easy_ocr_hindi,image_path)
            #     pool2.close()
            # except Exception as e:
            #     print(e)


            # image_path_house = [f"house_number/unorder_output_image_number{j}/cropped_table_{i}.png"
            #               for j in range(0, count)
            #               if count_file(f'house_number/unorder_output_image_number{j}') > 0
            #               for i in range(count_files(f'house_number/unorder_output_image_number{j}') - 1, -1, -1)]
            # try:
            #     for image in image_path_house:
            #         result = reader1.readtext(image)
            #         if result:
            #             text_result = result[0][1]
            #             data.append(text_result)
            #         else:
            #             text_result="M"
            #             data.append(text_result)
            # except Exception as e:
            #     print(e)
            #
            #
            # df_voter=pd.DataFrame({'house_number':data})
            # data_frame_voter.append(df_voter)
            #




            # #data1 = pool.map(extract_data, image_path)

            #
            #

            # pool3 = Pool()
            #
            # image_pathss = [f"images/unorder_output_image_{j}/cropped_table_{i}.png" for j in range(0, count) for i in
            #                 range(count_files(f'images/unorder_output_image_{j}') - 1, -1, -1)]
            #
            # args_list = list(zip(image_path_house, image_pathss))
            #
            #
            #
            # data2 = pool3.map(extract_data2_hindi, args_list)
            #
            #
            #
            # df2 = pd.DataFrame(data2)
            #
            # pool3.close()
            # pool3.join()
            #
            #
            # #
            # # #df = pd.DataFrame(data2)
            # #
            # # # df=df['serial_numbers'].apply(lambda x: x.strip("\n",""))
            # # # df['serial_numbers'] = df['serial_numbers'].astype(str)
            # # #
            # # # # Now you can apply the strip method
            # # # df['serial_numbers'] = df['serial_numbers'].apply(lambda x: x.replace('[', '')
            # # #                                                   .replace(']', '')
            # # #                                                   .replace('\\n', '')
            # # #                                                   .replace("'", '')
            # # #                                                   .replace('|', '')
            # # #                                                   .strip())
            # # #
            # #
            # #
            # # # ai model
            # #
            # data_frames.append(df2)
            # # #data_voter=pd.DataFrame(data)
            #
            #
            #
            pool10 = Pool()



            imgpath = [f"home/ec2-user/output_image_{i}.png" for i in range(count)]


            imgpath2 = [f"home/ec2-user/unorder_output_image_{i}" for i in range(count)]


            args_list = list(zip(imgpath, imgpath2))

            results = pool10.map(pooling_stations_hindi, args_list)



            d3 = []
            d4 = []
            for result in results:

                d3.extend(result[0])
                d4.extend(result[1])

            #
            #     # df_column = df1.transpose()
            #
            #     #df_combined.to_excel('combined_station_hindi.xlsx', index=False)
            #
            pool10.close()
            pool10.join()

            #
            #


            data_dict = {

                'section': d3,
                'section_Number': d4

            }

            df_combined=pd.DataFrame(data_dict)

            #pattern2=pd.DataFrame()
            # df_combined['Voter_number']= df_combined['Voter_number'].astype(str)
            # pattern2['First_3_digits'] = df_combined['Voter_number'].apply(lambda x: x[:5])
            #
            # value_counts = pattern2['First_3_digits'].value_counts()
            #
            #
            # # Get the value with the highest count
            # data_pattern = value_counts.reset_index()
            # data_pattern.columns = ['Pattern', 'Count']
            # print(data_pattern)
            #
            #
            #
            # pattern.append(data_pattern)
            #
            #


            data_frames2.append(df_combined)


            shutil.rmtree('home/ec2-user/images')
            shutil.rmtree('home/ec2-user/voter_number')
            shutil.rmtree('home/ec2-user/house_number')

            dir2 = "home/ec2-user/pdf"
            shutil.rmtree(dir2)

        #     print("Total time spent:", counter+1)
        #final_df = pd.concat(data_frames, ignore_index=True)
        #final_df.to_excel('combined_voter_hindi.xlsx', index=False)
        final_df2=pd.concat(data_frames2,ignore_index=True)
        final_df2.to_excel('combined_polling_hindi.xlsx', index=False)
        # final_df3 = pd.concat(data_frame_voter, ignore_index=True)
        # final_df3.to_excel('house_number_missing.xlsx', index=False)
        shutil.rmtree('home/ec2-user/data')
        os.makedirs('home/ec2-user/data',exist_ok=True)

        # df_combined=pd.read_excel(r"D:\ocr_production_hosting\epic_number.xlsx")
        #
        # pattern2 = pd.DataFrame()
        # df_combined['Voter_number']= df_combined['Voter_number'].astype(str)
        # pattern2['First_3_digits'] = df_combined['Voter_number'].apply(lambda x: x[:3])
        #
        # value_counts = pattern2['First_3_digits'].value_counts()
        #
        #
        # # Get the value with the highest count
        # data_pattern = value_counts.reset_index()
        # data_pattern.columns = ['Pattern', 'Count']
        #
        #
        # #
        #
        # data_pattern.to_excel('pattern_list.xlsx', index=False)
        end = datetime.now()
        print("total time for  processing", end - start)

        #
        # # dir2 = "D:/ocr_production_hosting/pdf"
        # # shutil.rmtree(dir2)
        # #
        # # shutil.rmtree('D:/ocr_production_hosting/images')
        # # shutil.rmtree('D:/ocr_production_hosting/voter_number')

    return jsonify({"message": "Files successfully uploaded with ocr", "file_paths": file_paths})






if __name__ == "__main__":
    app.run(debug=True,port=5001)


