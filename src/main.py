import sys
from typing import Optional, List, Tuple
from paddleocr import PaddleOCR
import cv2
from editdistance import distance as edit_distance
from logging import getLogger

lite_ocr = PaddleOCR(
    lang='ch',
    det_model_dir='../models/ch_PP-OCRv4_det_infer',
    rec_model_dir='../models/ch_PP-OCRv4_rec_infer',
    use_gpu=True
)
full_ocr = PaddleOCR(
    # use_angle_cls=True, # 不用方向检测了
    lang='ch',
    det_model_dir='../models/ch_PP-OCRv4_det_server_infer',
    rec_model_dir='../models/ch_PP-OCRv4_rec_server_infer',
    use_gpu=True
)
# 请先下载模型！https://paddlepaddle.github.io/PaddleOCR/main/ppocr/model_list.html

logger = getLogger(__name__)

def ocr_lite(frame) -> Optional[str]:
    """
    使用 paddleocr 轻量模型来检测和识别文本
    FIXME: 仅保留首个文本框的文本，画面中的文字将阻碍字幕识别！
    FIXME: 可能可以利用之前出现过的字幕限制当前字幕框
    TODO: 考虑利用返回的坐标来识别字幕位置，从而在下方提供准确的双语字幕
    :param frame:
    :return:

    tip: example output:
    a. [None]
    b. [[ line[] ]]
        line: [ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ('text', confidence) ]
    """
    result = lite_ocr.ocr(frame, cls=False)
    if result[0] is None:
        return None
    else:
        text0 = result[0][0][-1][0]
        return text0


def ocr_full(frame) -> Optional[str]:
    """
    使用 paddleocr 全量模型来检测和识别文本

    :param frame:
    :return:

    """
    result = full_ocr.ocr(frame, cls=False)
    if result[0] is None:
        return None
    else:
        text0 = result[0][0][-1][0]
        return text0


class Config:
    NEAR_TOLERANCE = 2  # 编辑距离，编辑距离小于该值的字幕将被认为是相似的
    DETECT_FREQ_FRAME = -1  # 每隔多少帧检测一次字幕，比下面优先生效，估测默认值：6
    DETECT_FREQ_S = 0.1  # 每隔多秒检测一次字幕，估测默认值：100 （标准播音员语速是每分钟 150 字左右，估计单字在 1/3 秒左右）
    PER_CHAR_JUMP = 0.1  # 每个字的跳跃时间，单位秒，估测默认值：0.1
    # 本代码基于中文构建！如果您使用其他语言，len(text) 将会非常大从而导致跳跃时间非常严重！此时请考虑置0或修改跳跃相关的代码，在find_change里

class VideoDetector:

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_gap = -1
        if Config.DETECT_FREQ_FRAME != -1:
            self.frame_gap = Config.DETECT_FREQ_FRAME
        elif Config.DETECT_FREQ_S != -1:
            self.frame_gap = int(self.fps * Config.DETECT_FREQ_S)
        else:
            self.frame_gap = 6 # 默认值

    @staticmethod
    def is_different(text1: str | None, text2: str | None) -> bool:
        """
        检查两个文本是否不同
        :param text1: 文本1
        :param text2: 文本2。文本2更重要，我们会考虑文本2是短文本的可能性
        :return:
        """
        if (text1 is None) ^ (text2 is None):
            return True
        elif text1 is None and text2 is None:
            return False
        elif edit_distance(text1, text2) > Config.NEAR_TOLERANCE:
            return True
        else:
            # 考虑较短的小段字幕的情况，比如好、嗯、哦、等等
            if len(text1) < Config.NEAR_TOLERANCE and len(text2) < Config.NEAR_TOLERANCE and not text2.isalnum():
                return True
            else:
                return False

    def find_subtitle_boundary(self, start_frame: int, end_frame: int, prev_subtitle: str,
                               precision=2):
        """
        找出字幕的分界点。
        由于字幕总是连续的，所以采用二分法来查找。

        :param start_frame: prev_text 还在的帧
        :param end_frame: prev_text 没在的帧
        :param prev_subtitle: 上一个字幕的文本
        :param precision: 帧精度，头 尾 差小于 precision 时将直接返回
        :return:
        """
        if end_frame - start_frame <= precision:
            return start_frame

        mid = int((start_frame + end_frame) // 2)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError('? should not happen')
        text = ocr_lite(frame) or ""
        if self.is_different(prev_subtitle, text):
            # 继续向前
            return self.find_subtitle_boundary(start_frame, mid, prev_subtitle, precision)
        else:
            # 继续向后
            return self.find_subtitle_boundary(mid, end_frame, prev_subtitle, precision)

    def find_subtitle_change(self, current_frame: int, this_subtitle: str):
        """
        向后跳跃，尝试寻找字幕已发生转换的帧

        :param current_frame: 当前字幕所在的帧
        :param this_subtitle: 当前字幕的文本

        """

        test_end = current_frame + len(this_subtitle) * self.fps  # 每个字0.1秒

        while True:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, test_end)
            ret, frame = self.video.read()
            if not ret:
                # 大概是到达视频末尾了
                test_end = self.total_frame_count  # 放到最后一帧，防死
                break
            tail_text = ocr_lite(frame)
            if tail_text is None or self.is_different(this_subtitle, tail_text):
                self.video.set(cv2.CAP_PROP_POS_FRAMES, test_end + self.frame_gap)  # 多扫一次
                ret, frame = self.video.read()
                if not ret:
                    test_end = self.total_frame_count
                    break
                tail_text_2 = ocr_lite(frame) # 多扫一次
                if tail_text_2 is None or self.is_different(this_subtitle, tail_text_2):
                    # 确实是结束了
                    # print(f'! {tail_text_2}, {tail_text}, {this_subtitle}')
                    break
                else:
                    # 小偏差，继续扫
                    test_end += self.frame_gap
            test_end += self.frame_gap
        return test_end

    def execute(self)-> List[Tuple[int, int, str]]:
        """
        主逻辑，执行字幕检测
        :return:
        """
        prev_frame_no = 0  # 扫完的帧
        current_frame_no = 0  # 正准备扫的帧
        prev_subtitle: str | None = None

        subtitle_result = []

        while self.video.isOpened():
            self.video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no)
            ret, frame = self.video.read()
            if not ret:
                print('end')
                break

            # main process

            current_subtitle_lite = ocr_lite(frame)
            if current_subtitle_lite is None:
                prev_frame_no = current_frame_no
                current_frame_no += self.frame_gap
                prev_subtitle = None
                continue
            elif self.is_different(prev_subtitle, current_subtitle_lite):
                current_subtitle_full = ocr_full(frame)
                if current_subtitle_full is None:
                    # 轻量模型识别的结果是误扫
                    prev_frame_no = current_frame_no
                    current_frame_no += self.frame_gap
                    prev_subtitle = ""
                    continue

                current_subtitle_begin = self.find_subtitle_boundary(prev_frame_no, current_frame_no, prev_subtitle)

                jump_end = self.find_subtitle_change(current_frame_no, current_subtitle_full)
                current_subtitle_end = self.find_subtitle_boundary(jump_end - self.frame_gap, jump_end, current_subtitle_lite)

                subtitle_result.append((current_subtitle_begin, current_subtitle_end, current_subtitle_full))

                logger.info(f'[{current_subtitle_begin} - {current_subtitle_end}] {current_subtitle_full}')

                prev_frame_no = current_subtitle_end + 1
                current_frame_no = current_subtitle_end + 2
                prev_subtitle = current_subtitle_full or ""
            elif len(current_subtitle_lite) > Config.NEAR_TOLERANCE:
                # FIXME: 不应该发生的情况，理论上，上面应该已经找到字幕转换点了？
                logger.warning(f'Found a similar subtitle: {prev_subtitle} -> {current_subtitle_lite}, is it a mistake?')
            else:
                # 轻量模型识别的结果是误扫，跳过
                prev_frame_no = current_frame_no
                current_frame_no += self.frame_gap
                prev_subtitle = ""
                logger.info(f'proceed: {current_frame_no} -> {self.total_frame_count}')

        self.video.release()
        return subtitle_result





if __name__ == '__main__':
    detector = VideoDetector('../input/test.mp4')
    print(detector.execute())