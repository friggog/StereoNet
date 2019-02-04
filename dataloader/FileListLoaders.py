import torch.utils.data as data

from PIL import Image
import os
import os.path
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def SceneFlowList(filepath):

    classes = [d for d in os.listdir(
        filepath) if os.path.isdir(os.path.join(filepath, d))]

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_right_disp = []

    # ========================= monkaa =======================

    monkaa_path = filepath+'monkaa/frames_cleanpass'
    monkaa_dir = os.listdir(monkaa_path)
    monkaa_disp = filepath+'monkaa/disparity'

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
                all_left_disp.append(monkaa_disp+'/'+dd +
                                     '/left/'+im.split(".")[0]+'.pfm')

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)
                all_right_disp.append(
                    monkaa_disp+'/'+dd+'/right/'+im.split(".")[0]+'.pfm')

    # ========================= flying =======================

    # flying_dir = filepath+'flying/TRAIN/frames_cleanpass'
    # flying_disp = filepath+'flying/TRAIN/disparity'
    # subdir = ['A','B','C']

    # for ss in subdir:
    #    flying = os.listdir(flying_dir+ss)

    #    for ff in flying:
    #        imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #        for im in imm_l:
    #            if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #                all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

    #            all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
    #            all_right_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/right/'+im.split(".")[0]+'.pfm')

    #            if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #                all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    # flying_dir = filepath+'flying/TEST/frames_cleanpass'
    # flying_disp = filepath+'flying/TEST/disparity'

    # subdir = ['A','B','C']

    # for ss in subdir:
    #    flying = os.listdir(flying_dir+ss)

    #    for ff in flying:
    #        imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
    #        for im in imm_l:
    #            if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
    #                test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

    #            test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')
    #            test_right_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/right/'+im.split(".")[0]+'.pfm')

    #            if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
    #                test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    # ========================= driving =======================

    driving_dir = filepath + 'driving/frames_cleanpass'
    driving_disp = filepath + 'driving/disparity'

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir+'/'+i+'/'+j+'/'+k+'/left/')
                for im in imm_l:
                    if is_image_file(driving_dir+'/'+i+'/'+j+'/'+k+'/left/'+im):
                        all_left_img.append(
                            driving_dir+'/'+i+'/'+j+'/'+k+'/left/'+im)
                    all_left_disp.append(
                        driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

                    if is_image_file(driving_dir+'/'+i+'/'+j+'/'+k+'/right/'+im):
                        all_right_img.append(
                            driving_dir+'/'+i+'/'+j+'/'+k+'/right/'+im)
                    all_right_disp.append(
                        driving_disp+'/'+i+'/'+j+'/'+k+'/right/'+im.split(".")[0]+'.pfm')

    random.Random(4).shuffle(all_left_img)
    random.Random(4).shuffle(all_right_img)
    random.Random(4).shuffle(all_left_disp)
    random.Random(4).shuffle(all_right_disp)

    n_test = 1500
    test_left_img = all_left_img[:n_test]
    all_left_img = all_left_img[n_test:]
    test_right_img = all_right_img[:n_test]
    all_right_img = all_right_img[n_test:]
    test_left_disp = all_left_disp[:n_test]
    all_left_disp = all_left_disp[n_test:]
    test_right_disp = all_right_disp[:n_test]
    all_right_disp = all_right_disp[n_test:]

    # TEMP
    all_left_img = all_left_img[:11562]
    all_right_img = all_right_img[:11562]
    all_left_disp = all_left_disp[:11562]
    all_right_disp = all_right_disp[:11562]

    assert(len(all_left_img) == len(all_right_img) ==
           len(all_left_disp) == len(all_right_disp))
    assert(len(test_left_img) == len(test_right_img) ==
           len(test_left_disp) == len(test_right_disp))

    print('Loaded ', len(all_left_img), len(test_left_img))

    return all_left_img, all_right_img, all_left_disp, all_right_disp, test_left_img, test_right_img, test_left_disp, test_right_disp


def KittiList(filepath):

    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    disp_R = 'disp_occ_1/'

    if '15' in filepath:
        train = os.listdir(filepath + 'training/' + left_fold)
        train = [filepath + 'training/' + left_fold + p for p in train]
        val = os.listdir(filepath + 'testing/' + left_fold)
        val = [p for p in val if '_10' in p]
        val = [filepath + 'testing/' + left_fold + p for p in val]
    else:
        image = os.listdir(filepath + left_fold)
        image = [filepath + left_fold + p for p in val]
        train = image[:200]
        val = image[200:]

    left_train = train
    right_train = [p.replace(left_fold, right_fold) for p in train]
    left_val = val
    right_val = [p.replace(left_fold, right_fold) for p in val]

    if os.path.exists(train[0].replace(left_fold, disp_L)):
        disp_train_L = [p.replace(left_fold, disp_L) for p in train]
        disp_train_R = [p.replace(left_fold, disp_R) for p in train]
        disp_val_L = [p.replace(left_fold, disp_L) for p in val]
        disp_val_R = [p.replace(left_fold, disp_R) for p in val]
    else:
        disp_train_L = ['' for _ in train]
        disp_train_R = ['' for _ in train]
        disp_val_L = ['' for _ in val]
        disp_val_R = ['' for _ in val]

    return left_train, right_train, disp_train_L, disp_train_R, left_val, right_val, disp_val_L, disp_val_R
