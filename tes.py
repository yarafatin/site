import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr


def process(img_path, lang):
    img_path = '/mnt/d/yaraf/hqocr/data/foreign/korean/00000005.tif'
    lang = 'korean'

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    # thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    # result = 255 - close

    # Remove horizontal
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    # detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(image, [c], -1, 0, 3)
    #
    # # Remove vertical
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    # detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(image, [c], -1, 0, 3)

    linek = np.zeros((11, 11), dtype=np.uint8)
    linek[5, ...] = 1
    x = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek, iterations=1)
    gray -= x

    cv2.imshow('result', gray)
    cv2.waitKey()
    result = gray

    # # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    # need to run only once to download and load model into memory
    # ocr = PaddleOCR(use_angle_cls=True, lang='en')
    ocr = PaddleOCR(lang=lang)
    result = ocr.ocr(result, det=True, cls=False)

    # print(result)
    # result.sort(key=lambda x: x[0][0][1])

    # for line in result:
    #     print(line)

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

    with open('{}.paddle.txt'.format(img_path), 'w') as fp:
        for key in myDict:
            myDict[key].sort(key=lambda x: x[1])
            data = ' '.join([tup[0] for tup in myDict[key]])
            fp.write("%s\n" % data)

    # with open(r'out.txt', 'w') as fp:
    #     for line in result:
    #         # write each item on a new line
    #         fp.write("%s\n" % line[1][0])
    #     print('Done')

    # draw result
    # from PIL import Image
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='./korean.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')

process('', '')