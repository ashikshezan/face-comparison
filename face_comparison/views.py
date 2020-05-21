import cv2
import face_recognition
import numpy as np
import datetime
from django.shortcuts import render
from django.http import JsonResponse
# Create your views here.


def home(request, *args, **kwargs):

    context = {
        'result': 'This is my result'
    }
    return render(request=request, template_name='home.html', context=context)


def compare_images(request, *args, **kwargs):
    if request.POST:
        img1 = request.FILES['image1']
        compare_face(img1)
        print(type(img1))

    else:
        print('Not a valid one')

    return JsonResponse({}, status=200)


def compare_face(img):

    results = []

    # upload images from web app
    # here just load from disk

    # input image 1
    imfile_name_1 = img
    actim1 = face_recognition.load_image_file(imfile_name_1)
    resizer1 = round(512/max(actim1.shape[0], actim1.shape[1]), 3)
    actim1_inp = cv2.resize(actim1, (0, 0), fx=resizer1, fy=resizer1)
    face_locs1 = face_recognition.face_locations(actim1_inp, model='hog')

    # input image 2
    imfile_name_2 = img
    actim2 = face_recognition.load_image_file(imfile_name_2)
    resizer2 = round(512/max(actim2.shape[0], actim2.shape[1]), 3)
    actim2_inp = cv2.resize(actim2, (0, 0), fx=resizer2, fy=resizer2)
    face_locs2 = face_recognition.face_locations(actim2_inp, model='hog')

    # Error messages top up on screen:
    error_messages_no = 0
    # if there's no face on any of the images
    if len(face_locs1) == 0 or len(face_locs2) == 0:
        error_messages_no += 1
        print('No face found on: ' +
              ('image 1' if len(face_locs1) == 0 else '') +
              (', ' if len(face_locs1) + len(face_locs2) == 0 else '') +
              ('image 2' if len(face_locs2) == 0 else ''))
    # if there are more then 1 face on any of the images
    if len(face_locs1) > 1 or len(face_locs2) > 1:
        error_messages_no += 1
        print('More then one faces found on: ' +
              ('image 1' if len(face_locs1) > 1 else '') +
              (', ' if (len(face_locs1) - 1) * (len(face_locs2) - 1) > 0 else '') +
              ('image 2' if len(face_locs2) > 0 else ''))

    score = '0.000'
    if error_messages_no == 0:
        ### MAIN calculation ###
        actenc1 = face_recognition.face_encodings(actim1_inp, face_locs1)
        actenc2 = face_recognition.face_encodings(actim2_inp, face_locs2)[0]
        match_flag = face_recognition.compare_faces(actenc1, actenc2)
        face_dist0 = face_recognition.face_distance(actenc1, actenc2)
        score = str(np.round(1 / (1 + np.exp(-8*(1-face_dist0[0]-0.4))), 3))

        # sending back result:
        ms = 'Not '
        if match_flag == True:
            ms = ''
        print('Matching %sFound - Matching Score: %s'
              % (ms, score))

    # Storing inputs and results (always, not only whithout error messages)
    # storing info and images whereever it's reasonable
    results.append([str(datetime.datetime.now().time()), imfile_name_1, imfile_name_2,
                    len(face_locs1), len(face_locs2), score])

    return score
