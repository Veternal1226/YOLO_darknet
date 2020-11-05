import time
import random
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import darknet as dn
import simpleaudio as sa

# prepare YOLO
net = dn.load_net(str.encode("cfg/yolov2-tiny.cfg"),
                  str.encode("weights/tiny-yolo.weights"), 0)
meta = dn.load_meta(str.encode("cfg/coco.data"))

# box colors
box_colors = None


def generate_colors(num_classes):
    global box_colors

    if box_colors != None and len(box_colors) > num_classes:
        return box_colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    box_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    box_colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            box_colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    # Shuffle colors to decorrelate adjacent classes.
    random.shuffle(box_colors)
    random.seed(None)  # Reset seed to default.


def draw_boxes(img, result):
    classArr = []

    image = Image.fromarray(img)

    font = ImageFont.truetype(font='NotoSansCJK-Medium.ttc', size=20)
    
    thickness = (image.size[0] + image.size[1]) // 300

    num_classes = len(result)
    generate_colors(num_classes)

    index = 0
    for objection in result:
        index += 1
        class_name, class_score, (x, y, w, h) = objection
        # print(name, score, x, y, w, h)
        if(class_name.decode('utf-8') != 'person'):
            continue
        left = int(x - w / 2)
        right = int(x + w / 2)
        top = int(y - h / 2)
        bottom = int(y + h / 2)

        label = '{} {:.2f}'.format(class_name.decode('utf-8'), class_score)
        classArr.append(class_name.decode('utf-8'))

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i,
                            bottom - i], outline=box_colors[index - 1])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=box_colors[index - 1])
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        del draw
    # draw person count backscreen
    drawBlk = ImageDraw.Draw(image,'RGBA')
    fontCount = ImageFont.truetype(font='NotoSansCJK-Medium.ttc', size=40)
    drawBlk.rectangle([0,10,400,50],fill=(0,0,0,int(255*0.5))) #black+opacity(0.5)=grey
    del drawBlk
    # draw person count number
    draw = ImageDraw.Draw(image)
    draw.text(np.array([0,0]), str('目前在場人數：　')+str(classArr.count('person')), fill=(255, 255, 255), font=fontCount)
    # if person count too many (4) , draw notice message
    if(classArr.count('person')>0):
        draw.rectangle([0,50,480,90],fill=(0,0,255))
        draw.text(np.array([0,40]), str('請保持社交距離或戴上口罩'), fill=(255, 255, 255), font=fontCount)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    del draw
    #for detectedObject in classArr:
    #    draw = ImageDraw.Draw(image)
    #    fontObj = ImageFont.truetype(font='NotoSansCJK-Medium.ttc', size=30)
    #    draw.text(np.array([0,40]), classArr[classArr.index(detectedObject)], fill=(255, 255, 255), font=fontObj)
    #    del draw
    return np.array(image),classArr


def array_to_image(arr):
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr / 255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w, h, c, data)
    return im


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    #im = dn.load_image(image, 0, 0)
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    dn.predict_image(net, image)
    dets = dn.get_network_boxes(net, image.w, image.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #dn.free_image(image)
    dn.free_detections(dets, num)
    return res


def pipeline(img):
    # image data transform
    # img - cv image
    # im - yolo image
    im = array_to_image(img)
    dn.rgbgr_image(im)

    tic = time.time()
    result = detect(net, meta, im)
    toc = time.time()
    print(toc - tic, result)

    img_final,cArr = draw_boxes(img, result)
    return img_final,cArr


count_frame, process_every_n_frame = 0, 1
# get camera device
cap = cv2.VideoCapture(0)
wave_obj = sa.WaveObject.from_wave_file("./mask.wav")

while(True):
    # get a frame
    ret, frame = cap.read()
    count_frame += 1

    # show a frame
    img = cv2.resize(frame, (0, 0), fx=1, fy=1)  # resize image half
    #cv2.imshow("Video", img)

    # if running slow on your computer, try process_every_n_frame = 10
    if count_frame % process_every_n_frame == 0:
        tmp, arr = pipeline(img)
        cv2.imshow("YOLO", tmp)
        #arr.append(str(arr.count('person')) + "")
        #with open("label.txt","a") as fs:
            #fs.write(",".join(arr))

    # press keyboard 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()