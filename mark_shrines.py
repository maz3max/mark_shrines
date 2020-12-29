#!/usr/bin/python3
import cv2 as cv
import numpy as np

from pycpd import AffineRegistration

import argparse
import os
import sys

# data is from https://github.com/zeldadungeon/maps/blob/master/src/botw/markers/pins.json
towers = [([3000, 6616], 'Akkala Tower'),
          ([-884, -1577], 'Central Tower'),
          ([-3428, 2034], 'Dueling Peaks Tower'),
          ([3114, 4348], 'Eldin Tower'),
          ([-6547, 2662], 'Faron Tower'),
          ([-3657, -7332], 'Gerudo Tower'),
          ([-3390, -1120], 'Great Plateau Tower'),
          ([-4267, 5471], 'Hateno Tower'),
          ([4068, -4346], 'Hebra Tower'),
          ([-5923, -64], 'Lake Tower'),
          ([218, 4516], 'Lanayru Tower'),
          ([1549, -3511], 'Ridgeland Tower'),
          ([1980, -7227], 'Tabantha Tower'),
          ([-4875, -4614], 'Wasteland Tower'),
          ([3211, 1768], 'Woodland Tower')]

shrines = [([-6923, -3590], 'Dila Maag Shrine'),
           ([-3310, 174], 'Bosh Kala Shrine'),
           ([-3980, -901], 'Ja Baij Shrine'),
           ([-4635, -1850], 'Owa Daim Shrine'),
           ([-3980, -2865], 'Keh Namut Shrine'),
           ([-3031, -1341], 'Oman Au Shrine'),
           ([-3849, 3329], 'Ha Dahamar Shrine'),
           ([-1681, 1704], 'Hila Rao Shrine'),
           ([-3702, 2496], 'Shee Venath Shrine'),
           ([-3877, 2538], 'Shee Vaneer Shrine'),
           ([-3689, 2551], 'Ree Dahee Shrine'),
           ([-1786, 3679], "Ta'loh Naeg Shrine"),
           ([-1941, 4075], 'Lakna Rokee Shrine'),
           ([-4945, 3698], 'Toto Sah Shrine'),
           ([-4426, 6771], 'Myahm Agana Shrine'),
           ([-764, 5243], 'Mezza Lo Shrine'),
           ([-3372, 8371], "Tahno O'ah Shrine"),
           ([-5976, 8019], 'Chaas Qeta Shrine'),
           ([-2633, 7771], "Jitan Sa'mi Shrine"),
           ([-2663, 5389], "Dow Na'eh Shrine"),
           ([-2995, 5007], 'Kam Urog Shrine'),
           ([1031, 6644], "Ne'ez Yohma Shrine"),
           ([-810, 6667], 'Rucco Maag Shrine'),
           ([582, 4483], 'Soh Kofi Shrine'),
           ([2615, 9414], 'Kah Mael Shrine'),
           ([-374, 1656], 'Kaya Wan Shrine'),
           ([756, 3014], 'Sheh Rata Shrine'),
           ([-929, 3208], 'Daka Tuss Shrine'),
           ([-513, 8489], 'Shai Yota Shrine'),
           ([838, 6293], 'Dagah Keek Shrine'),
           ([-5628, -5383], "Korsh O'hu Shrine"),
           ([-5634, -7629], 'Daqo Chisay Shrine'),
           ([-2618, -3119], 'Rota Ooh Shrine'),
           ([-6567, 4021], 'Qukah Nata Shrine'),
           ([-6614, 7309], 'Muwo Jeem Shrine'),
           ([-1439, -1934], "Kaam Ya'tak Shrine"),
           ([1808, -4538], 'Toh Yahsa Shrine'),
           ([1181, -2867], 'Zalta Wa Shrine'),
           ([2952, -2982], 'Monya Toma Shrine'),
           ([-7045, 1044], "Ka'o Makagh Shrine"),
           ([6848, 6646], 'Zuna Kai Shrine'),
           ([-3407, -3394], 'Dah Kaso Shrine'),
           ([872, -5861], 'Shae Loya Shrine'),
           ([892, -6938], "Tena Ko'sah Shrine"),
           ([-7540, 9468], 'Korgu Chideh Shrine'),
           ([2613, 7800], 'Dah Hesho Shrine'),
           ([4262, 9052], 'Ritaag Zumo Shrine'),
           ([5457, 8585], 'Katosa Aug Shrine'),
           ([7412, 9310], "Tu Ka'loh Shrine"),
           ([5403, 7551], 'Tutsuwa Nima Shrine'),
           ([3337, 6062], 'Ze Kasho Shrine'),
           ([2325, 5444], "Mo'a Keet Shrine"),
           ([7070, -1649], 'Qaza Tokki Shrine'),
           ([7521, -3340], 'Sha Gehma Shrine'),
           ([-1811, -9309], 'Kema Kosassa Shrine'),
           ([3521, -7312], "Akh Va'quot Shrine"),
           ([4653, 4125], 'Daqa Koh Shrine'),
           ([5119, 3509], "Shae Mo'sah Shrine"),
           ([6239, 574], 'Ketoh Wawai Shrine'),
           ([-4601, -5614], 'Kay Noh Shrine'),
           ([-4326, -6628], 'Dako Tah Shrine'),
           ([4515, -4764], "Gee Ha'rah Shrine"),
           ([-2020, 695], 'Wahgo Katta Shrine'),
           ([7424, -8039], 'To Quomo Shrine'),
           ([-3913, -7123], 'Sasa Kai Shrine'),
           ([7601, -8888], 'Hia Miu Shrine'),
           ([1643, 1515], 'Namika Ozz Shrine'),
           ([3884, 41], 'Kuhn Sidajj Shrine'),
           ([4330, 943], 'Keo Ruug Shrine'),
           ([4910, -50], 'Daag Chokah Shrine'),
           ([4833, 1671], 'Maag Halan Shrine'),
           ([2425, 2456], 'Mirro Shaz Shrine'),
           ([3041, 3638], 'Qua Raym Shrine'),
           ([6232, 3077], 'Shora Hah Shrine'),
           ([2319, -302], "Saas Ko'sah Shrine"),
           ([683, -1270], 'Katah Chuki Shrine'),
           ([1249, -1910], 'Noya Neha Shrine'),
           ([-7223, 3171], 'Shai Utoh Shrine'),
           ([-5991, 3579], 'Shoda Sah Shrine'),
           ([-5988, 1118], 'Pumaag Nitae Shrine'),
           ([-7682, 196], 'Shoqa Tatone Shrine'),
           ([-5199, -664], 'Ya Naga Shrine'),
           ([-4665, 1742], 'Shae Katha Shrine'),
           ([-7130, -1977], 'Ishto Soh Shrine'),
           ([-6903, -2833], 'Suma Sahma Shrine'),
           ([5317, -2181], 'Rona Kachta Shrine'),
           ([3158, 5337], 'Sah Dahaj Shrine'),
           ([6918, 5329], 'Gorae Torr Shrine'),
           ([1706, 8391], "Ke'nai Shakah Shrine"),
           ([-6640, 6877], 'Kah Yah Shrine'),
           ([-6629, 5668], 'Yah Rin Shrine'),
           ([-5665, 5269], 'Tawa Jinn Shrine'),
           ([-7556, -5937], 'Misae Suma Shrine'),
           ([-6252, -7614], 'Raqa Zunzo Shrine'),
           ([-3356, -4008], 'Joloo Nah Shrine'),
           ([-2448, -6171], 'Kuh Takkar Shrine'),
           ([-3302, -7818], 'Sho Dantu Shrine'),
           ([-3943, -9347], 'Kema Zoos Shrine'),
           ([-5599, -9591], 'Tho Kayu Shrine'),
           ([-7552, -9697], 'Hawa Koth Shrine'),
           ([-929, -4598], 'Mogg Latan Shrine'),
           ([-179, -3780], 'Sheem Dagoze Shrine'),
           ([-448, -5492], 'Mijah Rokee Shrine'),
           ([-1431, -7700], 'Keeha Yoog Shrine'),
           ([827, -8248], 'Kah Okeo Shrine'),
           ([2923, -3884], "Maag No'rah Shrine"),
           ([3149, -5662], 'Dunba Taag Shrine'),
           ([3038, -7217], 'Bareeda Naag Shrine'),
           ([3448, -8026], 'Voo Lota Shrine'),
           ([4417, -7640], 'Sha Warvo Shrine'),
           ([5016, -8123], 'Maka Rah Shrine'),
           ([6077, -7248], 'Mozo Shenno Shrine'),
           ([6436, -5995], 'Shada Naw Shrine'),
           ([5763, -5592], 'Goma Asaagh Shrine'),
           ([6449, -4748], 'Rok Uwog Shrine'),
           ([4128, -5276], 'Lanno Kooh Shrine'),
           ([5110, -3435], 'Rin Oyaa Shrine'),
           ([-4840, -3589], 'Jee Noh Shrine'),
           ([1878, 4607], 'Tah Muhl Shrine'),
           ([4079, 4146], 'Kayra Mah Shrine')]

parser = argparse.ArgumentParser(
    description='Marks shrine locations in a screenshot of Zelda: Breath of the Wild.')
parser.add_argument('image',
                    metavar='I',
                    type=str,
                    help='the screenshot image file')
args = parser.parse_args()
if os.path.isfile(args.image):
    img_rgb = cv.imread(args.image)
    output_name = os.path.splitext(args.image)[0] + "_with_markers.png"
else:
    sys.exit(-1)


# img_rgb = cv.imread('cropped.jpg')


# Step 1: find tower markers in screenshot
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('tower-template.png', 0)
w, h = template.shape[::-1]
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
threshold = 0.65
loc = np.where(res >= threshold)

# Step 2: transform known coordinates to locations found in screenshot


# This transforms the known coordinates into the screenshot coordinate system.
# The default transform works for a fully zoomed-out map
def transformY(_y,
               B=np.array([[-4.50175990e-02,  5.61971462e-06],
                           [1.31336463e-05,  4.50075562e-02]]),
               t=np.array([270.70929346, 630.81663146])):
    return np.dot(_y, B) + np.tile(t, (_y.shape[0], 1))


# format source and target points
X = np.array(list(zip(loc[0], loc[1])))
Y = np.array([t[0] for t in towers])
# apply default transform (to have a known good starting point)
Y = transformY(Y)

# run CPD algorithm to register point clouds and find precise transform
reg = AffineRegistration(**{'X': X, 'Y': Y})
reg.register()
# apply correction transform to tower locations
Y = transformY(Y, reg.B, reg.t)

# mark tower location in red
for pt in Y:
    pt = np.round(pt[::-1]+np.array([(w/2-0.5), h/2]))
    pt = (int(pt[0]), int(pt[1]))
    cv.circle(img_rgb, tuple(pt), w//2, (0, 0, 255), 2)

# format shrine locations
S = np.array([s[0] for s in shrines])
# apply default and corrective transform to shrine locations
S = transformY(S)
S = transformY(S, reg.B, reg.t)

# mark shrine locations in blue
for pt in S:
    pt = np.round(pt[::-1]+np.array([(w/2-0.5), (h/2+3)]))
    pt = (int(pt[0]), int(pt[1]))
    cv.circle(img_rgb, tuple(pt), w//2, (255, 0, 0), 2)

# save result image
cv.imwrite(output_name, img_rgb)
