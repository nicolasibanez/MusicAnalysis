import os
import sys


def main():
    path = 'https://drive.google.com/file/d/19xXeKczn_g_QShj4FJdaorJOzJOz0Yxa/view?usp=sharing'
    # download the file
    os.system('wget ' + path)
    # unzip the file
    # name : podcast001.tar
    os.system('tar -xvf podcast001.tar')



if __name__ == '__main__':
    main()