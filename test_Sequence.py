from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from network.model import TextScannerModel
from utils.visualise_callback import TBoardVisual
from utils.sequence import SequenceData
from utils.label import label_utils
from utils import logger as log
from utils import util
from utils.label.label_maker import LabelGenerater
import logging
import conf
import os
import time
import math
import numpy as np
import cv2
import scipy.ndimage.filters as fi
from utils import image_utils
from utils.label.label import ImageLabel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger("SequenceData")

class LabelGenerater():
    def __init__(self, max_sequence, target_image_shape, charset):
        self.max_sequence = max_sequence
        self.target_image_shape = target_image_shape  # [H,W]: [64,256]
        self.target_width = target_image_shape[1]
        self.target_height = target_image_shape[0]
        self.charset = charset

    def process(self, image_labels):

        # adjust the coordination
        shape = image_labels.image.shape[:2]  # h,w
        boxes = image_labels.bboxes  # [N,4,2] N: words number
        label = image_labels.label

        # # find the one bbox boundary
        # xmins = boxes[:, :, 0].min(axis=1)
        # xmaxs = np.maximum(boxes[:, :, 0].max(axis=1), xmins + 1)
        # ymins = boxes[:, :, 1].min(axis=1)
        # ymaxs = np.maximum(boxes[:, :, 1].max(axis=1), ymins + 1)

        character_segment = self.render_character_segemention(image_labels)
        localization_map = np.zeros(self.target_image_shape, dtype=np.float32)
        order_segments = np.zeros((*self.target_image_shape, self.max_sequence), dtype=np.float32)
        #order_maps = np.zeros((*self.target_image_shape, self.max_sequence), dtype=np.float32)

        assert boxes.shape[0] <= self.max_sequence, \
            f"the train/validate label text length[{len(image_labels.labels)}] must be less than pre-defined max sequence length[{self.max_sequence}]"

        # process each character
        for i in range(boxes.shape[0]):
            # Y_hat_k is the normalized_gaussian map, comply with the name in the paper
            Y_hat_k = self.generate_Y_hat_k_by_gaussian_normalize(self.target_image_shape,
                                                                  boxes[i])  # xmins[i], xmaxs[i], ymins[i], ymaxs[i])
            if Y_hat_k is None:
                logger.warning("Y_%d generator failed,the char[%s] of [%s]", i, label[i], label)
                Y_hat_k = np.zeros((self.target_image_shape))

            self.render_order_segment(order_segments[:, :, i], Y_hat_k, threshold=self.ζ)
            localization_map = self.render_localization_map(localization_map, Y_hat_k)
            #order_maps = order_segments * localization_map[:, :, np.newaxis]

        return character_segment, order_segments, localization_map

    # 围绕中心点做一个高斯分布，但是由于每个点的概率值过小，所以要做一个归一化,使得每个点的值归一化到[0,1]之间
    # Make a gaussian distribution with the center, and do normalization
    # def gaussian_normalize(self, shape, xmin, xmax, ymin, ymax)：
    # @return a "image" with shape[H,W], which is filled by a gaussian distribution
    def generate_Y_hat_k_by_gaussian_normalize(self, shape, one_word_bboxes):  # one_word_bboxes[4,2]
        # logger.debug("The word bbox : %r , image shape is : %r", one_word_bboxes, shape)

        # find the one bbox boundary
        xmin = one_word_bboxes[:, 0].min()
        xmax = one_word_bboxes[:, 0].max()
        ymin = one_word_bboxes[:, 1].min()
        ymax = one_word_bboxes[:, 1].max()

        out = np.zeros(shape)
        h, w = shape[:2]
        # find the "Center" of polygon
        y = (ymax + ymin + 1) // 2
        x = (xmax + xmin + 1) // 2
        if x > w or y > h:
            logger.warning("标注超出图像范围，生成高斯样本失败：(xmin:%f, xmax:%f, ymin:%f, ymax:%f,w:%f,x:%f,h:%f,y:%f)", xmin, xmax,
                           ymin, ymax, w, x, h, y)
            return None

        # prepare the gaussian distribution,refer to paper <<Label Generation>>
        out[y, x] = 1.

        fi.gaussian_filter(out, (self.δ, self.δ), output=out, mode='mirror')

        # logger.debug("Max gaussian value is :%f", out.max()) # it is 0.006367
        if out is None: return None

        return out

    def render_order_segment(self, order_maps, Y_k, threshold):
        Z_hat_k = Y_k / Y_k.max()
        Z_hat_k[Z_hat_k < threshold] = 0
        # Z_hat_k[Z_hat_k >= threshold] = 1
        order_maps[:] = Z_hat_k

    # fill the shrunk zone with the value of character ID
    def render_character_segemention(self, image_labels):

        character_segment = np.zeros(self.target_image_shape, dtype=np.int32)

        for one_word_label in image_labels.labels:
            label = one_word_label.label
            char_id = label_utils.str2id(label, self.charset)

            # shrink one word bboxes to avoid overlap
            shrinked_poly = image_utils.shrink_poly(one_word_label.bbox, self.shrink)

            word_fill = np.zeros(self.target_image_shape, np.uint32)

            word_fill.fill(char_id)

            mask = np.zeros(self.target_image_shape, np.uint8)
            cv2.fillPoly(mask, [shrinked_poly], 1)  # set those words' bbox area value to 1
            character_segment = np.maximum(mask * word_fill, character_segment) # merge two, only by maximum, not add
            character_segment.astype(np.int32)

        return character_segment

    # merge all Y_k with Max value
    def render_localization_map(self, localization_map, Y_k):
        return np.maximum(localization_map, Y_k)

class SequenceData(Sequence):
    def __init__(self, name, label_dir, label_file, charsets, conf, args, batch_size=32):
        self.conf = conf
        self.label_dir = label_dir
        self.name = name
        self.label_file = label_file
        self.batch_size = batch_size
        self.charsets = charsets
        self.initialize(args)
        self.start_time = time.time()
        self.target_image_shape = (conf.INPUT_IMAGE_HEIGHT, conf.INPUT_IMAGE_WIDTH)
        self.label_generator = LabelGenerater(conf.MAX_SEQUENCE, self.target_image_shape, charsets)

    def initialize(self, args):
        logger.info("[%s]begin to load image/labels", self.name)
        start_time = time.time()
        # 得到一个N*2的列表，两列分别是图像名和标签名
        self.data_list = label_utils.load_labels(self.label_dir)
        if len(self.data_list) == 0:
            msg = f"[{self.name}] 图像和标签加载失败[目录：{self.label_dir}]，0条！"
            raise ValueError(msg)

        logger.info("[%s]loaded [%d] labels,elapsed time [%d]s", self.name, len(self.data_list),
                    (time.time() - start_time))

    def __len__(self):
        return int(math.ceil(len(self.data_list) / self.batch_size))

    def load_image_label(self, batch_data_list):
        images = []
        batch_cs = []  # Character Segment
        batch_os = []  # Order Segment
        # batch_om = [] # Order Map
        batch_lm = []  # Localization Map
        label_text = []  # label text
        for image_path, label_path in batch_data_list:

            if not os.path.exists(image_path):
                logger.warning("Image [%s] does not exist", image_path)
                continue

            label_file = open(label_path, encoding="utf-8")
            data = label_file.readlines()
            label_file.close()
            logger.debug("Loaded label file [%s] %d lines", label_path, len(data))
            target_size = (self.target_image_shape[1], self.target_image_shape[0])
            il = ImageLabel(cv2.imread(image_path), data, self.conf.LABLE_FORMAT,
                            target_size=target_size)  # inside it, the bboxes size will be adjust
            logger.debug("Loaded label generates training labels")

            images.append(il.image)

            # text label
            label = il.label
            label_ids = label_utils.strs2id(label, self.charsets)
            label_text.append(label_ids)

            # character_segment, order_maps, localization_map = self.label_generator.process(il)
            character_segment, order_sgementation, localization_map = self.label_generator.process(il)
            character_segment = to_categorical(character_segment, num_classes=len(self.charsets) + 1)

            batch_cs.append(character_segment)
            # batch_om.append(order_maps)
            batch_os.append(order_sgementation)
            batch_lm.append(localization_map)

        images = np.array(images, np.float32)
        batch_cs = np.array(batch_cs)
        # batch_om = np.array(batch_om)
        batch_os = np.array(batch_os)
        batch_lm = np.array(batch_lm)

        # text one hot array
        labels = pad_sequences(label_text, maxlen=self.conf.MAX_SEQUENCE, padding="post", value=0)
        labels = to_categorical(labels, num_classes=len(self.charsets))

        # logger.debug("Loaded images:  %r", images.shape)
        # logger.debug("Loaded batch_cs:%r", batch_cs.shape)
        # logger.debug("Loaded batch_om:%r", batch_om.shape)
        # logger.debug("Loaded batch_lm:%r", batch_lm.shape)
        # logger.debug("[%s] loaded %d data", self.name, len(images))

        return images, [batch_cs, batch_os, batch_lm, labels]

    def __getitem__(self, idx):
        batch_data_list = self.data_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        images, labels = self.load_image_label(batch_data_list)
        print(images)
        return images, labels

if __name__ == '__main__':
    log.init()
    args = conf.init_args()
    charset = label_utils.get_charset(conf.CHARSET)
    train_sequence = SequenceData(name="Train",
                                  label_dir=args.train_label_dir,
                                  label_file=args.train_label_file,
                                  charsets=charset,
                                  conf=conf,
                                  args=args,
                                  batch_size=args.batch)

    for seq in train_sequence:
        print(seq)
