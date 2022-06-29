# import sys
# from os.path import dirname
# sys.path.append(dirname(__file__)+'/site-packages')

import cv2
from paddleocr import PaddleOCR
import pytesseract
import numpy as np
import os

work_dir = os.path.dirname(__file__)

det_dir_template = work_dir + "/models/det/{}/{}_PP-OCRv3_det_infer"
rec_dir_template = work_dir + "/models/rec/{}/{}_PP-OCRv3_rec_infer"
cls_dir = work_dir + "/models/cls/ch_ppocr_mobile_v2.0_cls_infer"

DEFAULT_DET_LANG = "ml"
det_langs = ("ch", "en", DEFAULT_DET_LANG)
DEFAULT_REC_LANG = "latin"
rec_langs = ("ch", "en", "japan", "korean", DEFAULT_REC_LANG)

TESSDATA_PATH = work_dir + "/tessdata"


def process(gray, lang):
    try:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        det_lang = lang if (lang in det_langs) else DEFAULT_DET_LANG
        rec_lang = lang if (lang in rec_langs) else DEFAULT_REC_LANG

        ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=False,
            det_model_dir=det_dir_template.format(det_lang, det_lang),
            rec_model_dir=rec_dir_template.format(rec_lang, rec_lang),
            cls_model_dir=cls_dir,
        )
        result = ocr.ocr(thresh, det=True, cls=False)

        myDict = {}
        prevline = 0
        for line in result:
            coord = line[0]
            key = coord[0][1]
            xaxis = coord[0][0]
            if prevline != 0:
                diff = key - prevline
                if diff <= 45:
                    myDict.setdefault(prevline, []).append((line[1][0], xaxis))
                else:
                    myDict.setdefault(key, []).append((line[1][0], xaxis))
                    prevline = key
            else:
                myDict.setdefault(key, []).append((line[1][0], xaxis))
                prevline = key

        ocr = []
        for key in myDict:
            myDict[key].sort(key=lambda x: x[1])
            ocr.append(" ".join([tup[0] for tup in myDict[key]]))
        return "\n".join(ocr)
    except BaseException as err:
        print(err)
        return "unable to ocr"


def initial(gray):
    try:
        # sharpen
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(
            sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        result = 255 - close

        # resize
        scale_percent = 70
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

        # ocr
        custom_config = "-l  chi_sim+kor+jpn --tessdata-dir " + TESSDATA_PATH
        return pytesseract.image_to_string(resized, config=custom_config)
    except BaseException as err:
        return "error in initial ocr"
