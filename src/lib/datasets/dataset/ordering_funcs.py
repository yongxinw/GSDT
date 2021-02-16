# Created by yongxinwang at 2020-08-16 18:45
import os


def identity(x):
    return x


def cuhksysu(x):
    return int(os.path.splitext(os.path.basename(x))[0][1:])


def caltech(x):
    return "{}_{}_{:06d}".format(
        *(os.path.splitext(os.path.basename(x))[0].split('_')[:2] +
          [int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])])
    )

