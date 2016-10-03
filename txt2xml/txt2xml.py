# -*- coding: utf-8 -*-
# author = sai

import xml.dom.minidom as xdm
import cv2
def to_xml(imagefile, tempplate = 'template.xml', save_xml='./Annotations/', save_txt='./ImageSets/Main/trainval.txt'):
    with open(imagefile, 'r') as f:
        for line in f.readlines():
            infos = line.split(' ')
            image = infos[0].split('\\')[1]
            path = infos[0].split('\\')[0] + '/' + image
            im = cv2.imread(path)
            width, height, depth = im.shape
            xmin = infos[1]
            xmax = infos[2]
            ymin = infos[3]
            ymax = infos[4]

            '''parse xml'''
            dom = xdm.parse(tempplate)
            root = dom.documentElement

            filename_dom = root.getElementsByTagName('filename')
            filename_dom[0].firstChild.data = image

            width_dom = root.getElementsByTagName('width')
            width_dom[0].firstChild.data = width

            height_dom = root.getElementsByTagName('height')
            height_dom[0].firstChild.data = height

            depth_dom = root.getElementsByTagName('depth')
            depth_dom[0].firstChild.data = depth

            xmin_dom = root.getElementsByTagName('xmin')
            xmin_dom[0].firstChild.data = xmin

            xmax_dom = root.getElementsByTagName('xmax')
            xmax_dom[0].firstChild.data = xmax

            ymin_dom = root.getElementsByTagName('ymin')
            ymin_dom[0].firstChild.data = ymin

            ymax_dom = root.getElementsByTagName('ymax')
            ymax_dom[0].firstChild.data = ymax
            xml_prefix = image.split('.')[0]
            xml_filename = save_xml + xml_prefix + '.xml'
            with open(xml_filename, 'w') as f:
                     dom.writexml(f, encoding='utf-8')
            with open(save_txt, 'a+') as f:
                     f.write(xml_prefix)
                     f.write('\n')

if __name__ == '__main__':
    imagefile = 'bioid_1471_bbox.txt'
    to_xml(imagefile)
