import glob
import logging
import os
from typing import Optional, List, Tuple

import cv2
import ffmpeg
import googletrans
from editdistance import distance as edit_distance
from paddleocr import PaddleOCR

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def ocr_lite(frame) -> Optional[Tuple[str, float]]:
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
        # entry0 = result[0][0][-1]
        # if entry0[1] >= 0.6:
        #     return entry0[0], entry0[1]  # text, confidence
        # elif len(result[0]) > 1:
        #     print(result)
        #     pass
        #     return None
        # else:
        #     return None
        if len(result[0]) == 1:
            entry0 = result[0][0][1]
            if entry0[1] >= 0.7:
                return entry0[0], entry0[1]
            else:
                return None
        else:
            rv = ""
            c = 0.0
            for _, entry in result[0]:
                if entry[1] >= 0.85:
                    rv += entry[0]
                    c = max(c, entry[1])
                elif entry[1] >= 0.6:
                    c = max(c, 0.7)
            if rv:
                return rv, c
            elif c < 0.8:
                return "", 0.7  # 提醒可能存在未出现完全的字幕
            else:
                return None


def ocr_full(frame) -> Optional[Tuple[str, float]]:
    """
    使用 paddleocr 全量模型来检测和识别文本

    :param frame:
    :return:

    """
    result = full_ocr.ocr(frame, cls=False)
    if result[0] is None:
        return None
    else:
        if len(result[0]) == 1:
            entry0 = result[0][0][1]
            return entry0[0], entry0[1]
        else:
            rv = ""
            c = 0.0
            for _, entry in result[0]:
                if entry[1] >= 0.85:
                    rv += entry[0]
                    c = max(c, entry[1])
                elif entry[1] >= 0.6:
                    c = max(c, 0.7)
            if rv:
                return rv, c
            elif c < 0.8:
                return "", 0.7  # 提醒可能存在未出现完全的字幕
            else:
                return None


class Config:
    NEAR_TOLERANCE = 2  # 编辑距离，编辑距离小于该值的字幕将被认为是相似的
    DETECT_FREQ_FRAME = -1  # 每隔多少帧检测一次字幕，比下面优先生效，估测默认值：6
    DETECT_FREQ_S = 0.1  # 每隔多少秒检测一次字幕，估测默认值：0.1 （标准播音员语速是每分钟 150 字左右，估计单字在 1/3 秒左右）
    PER_CHAR_JUMP = 0.1  # 每个字的跳跃时间，单位秒，估测默认值：0.1
    # 本代码基于中文构建！如果您使用其他语言，len(text) 将会非常大从而导致跳跃时间非常严重！此时请考虑置0或修改跳跃相关的代码，在find_change里


class VideoDetector:

    def __init__(self, video_path: str, output_dir: str = None):
        self.video_path = video_path
        self.output_dir = output_dir or "../output"
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_gap = -1
        if Config.DETECT_FREQ_FRAME != -1:
            self.frame_gap = int(Config.DETECT_FREQ_FRAME)
        elif Config.DETECT_FREQ_S != -1:
            self.frame_gap = int(self.fps * Config.DETECT_FREQ_S)
        else:
            self.frame_gap = 6  # 默认值
        if self.frame_gap <= 0:
            self.frame_gap = 1
        logger.info(f"init video detector({video_path}), fps: {self.fps}, total frame count: {self.total_frame_count},")

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
            pass
            return False
        elif edit_distance(text1, text2) > Config.NEAR_TOLERANCE:
            return True
        else:
            # 考虑较短的小段字幕的情况，比如好、嗯、哦、等等
            if len(text1) < Config.NEAR_TOLERANCE and len(text2) < Config.NEAR_TOLERANCE and not text2.isalnum():
                return True
            else:
                return False

    def frame_to_srt_time(self, frame_no: int) -> str:
        """
        将帧转换为 srt 时间格式
        :param frame_no: 帧数
        :return: srt 时间格式
        """
        seconds = int(frame_no / self.fps)
        milliseconds = int((frame_no % self.fps) * 1000 / self.fps)
        return f"{seconds // 3600:02}:{(seconds // 60) % 60:02}:{seconds % 60:02},{milliseconds:03}"

    def find_subtitle_boundary(self, start_frame: int, end_frame: int, prev_subtitle: str,
                               precision=2) -> int:
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
            return start_frame
            # print(mid, self.total_frame_count)
            # raise RuntimeError('? should not happen')
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

        test_end = int(current_frame + len(this_subtitle) * self.fps * Config.PER_CHAR_JUMP)  # 每个字0.1秒

        while True:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, test_end)
            ret, frame = self.video.read()
            if not ret:
                # 大概是到达视频末尾了
                test_end = self.total_frame_count  # 放到最后一帧，防死
                break
            tail_text = ocr_lite(frame)
            if tail_text is None or self.is_different(this_subtitle, tail_text[0]):
                break

                # self.video.set(cv2.CAP_PROP_POS_FRAMES, test_end + self.frame_gap)  # 多扫一次
                # ret, frame = self.video.read()
                # if not ret:
                #     test_end = self.total_frame_count
                #     break
                # tail_text_2 = ocr_lite(frame)  # 多扫一次
                # if tail_text_2 is None or self.is_different(this_subtitle, tail_text_2[0]):
                #     # 确实是结束了
                #     # print(f'! {tail_text_2}, {tail_text}, {this_subtitle}')
                #     break
                # else:
                #     # 小偏差，继续扫
                #     test_end += self.frame_gap

            test_end += self.frame_gap
        return test_end

    def subtitle_detect(self) -> List[Tuple[int, int, str]]:
        """
        核心逻辑1，执行字幕检测
        :return:
        """
        prev_frame_no = 0  # 扫完的帧
        current_frame_no = 0  # 正准备扫的帧
        prev_subtitle: str | None = None

        subtitle_result: List[Tuple[int, int, str]] = []

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
            elif self.is_different(prev_subtitle, current_subtitle_lite[0]):
                current_full_result = ocr_full(frame)
                if current_full_result is None or current_full_result[1] < 0.6:
                    # 轻量模型识别的结果是误扫
                    prev_frame_no = current_frame_no
                    current_frame_no += self.frame_gap
                    prev_subtitle = ""
                    continue
                elif current_full_result[1] < 0.8:
                    # 字幕未加载完全
                    prev_frame_no = current_frame_no
                    current_frame_no += 1  # 少跳一点
                    prev_subtitle = ""
                    continue
                current_subtitle: str = current_full_result[0]
                confidence: float = current_full_result[1]
                current_subtitle_begin = self.find_subtitle_boundary(prev_frame_no, current_frame_no, prev_subtitle)

                jump_end = self.find_subtitle_change(current_frame_no, current_subtitle)

                if current_frame_no > 100:
                    pass

                current_subtitle_end = self.find_subtitle_boundary(jump_end - self.frame_gap, jump_end,
                                                                   current_subtitle)

                subtitle_result.append((current_subtitle_begin, current_subtitle_end, current_subtitle))

                logger.info(f'[{current_subtitle_begin} - {current_subtitle_end}] {current_full_result}')
                logger.info(f'proceed: {current_frame_no} -> {self.total_frame_count}')

                prev_frame_no = current_subtitle_end + 1
                current_frame_no = current_subtitle_end + 2
                prev_subtitle = current_full_result or ""
            elif len(current_subtitle_lite[0]) > Config.NEAR_TOLERANCE:
                # FIXME: 不应该发生的情况，理论上，上面应该已经找到字幕转换点了？
                logger.warning(
                    f'Found a similar subtitle: {prev_subtitle} -> {current_subtitle_lite[0]}, is it a mistake?')
            else:
                # 轻量模型识别的结果是误扫，跳过
                prev_frame_no = current_frame_no
                current_frame_no += self.frame_gap
                prev_subtitle = ""

        self.video.release()
        return subtitle_result


    @staticmethod
    async def subtitle_translate(subtitle: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """
        核心逻辑2，执行字幕翻译
        :param subtitle:
        :return:
        """
        translator = googletrans.Translator()
        result = []
        # TODO: 为了保持上下文，我们可以多翻译几句字幕？
        for i, (start, end, text) in enumerate(subtitle):
            try:
                response = await translator.translate(text, src='zh-cn', dest='en')
                translated_text = response.text
                logger.info(translated_text)
                result.append((start, end, translated_text))
            except Exception as e:
                logger.error(f'Error translating {text}: {e}')

        return result

    def generate_srt(self, subtitle: List[Tuple[int, int, str]]) -> str:
        """
        生成 srt 字幕文本
        :param subtitle:
        :return:
        """
        srt = ""
        for i, (start, end, text) in enumerate(subtitle):
            srt += f"{i + 1}\r\n"  # srt seems to need crlf
            srt += f"{self.frame_to_srt_time(start)} --> {self.frame_to_srt_time(end)}\r\n"
            srt += f"{text}\r\n\r\n"
        return srt

    def save_srt(self, subtitle: List[Tuple[int, int, str]], filename: str):
        """
        保存 srt 字幕文本
        :param subtitle:
        :param filename:
        :return:
        """
        srt = self.generate_srt(subtitle)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(srt)
        logger.info(f'Saved srt to {filename}')

    def merge_soft_sub(self, subtitle: List[Tuple[int, int, str]]):
        """
        合并软字幕
        :param subtitle:
        :param filename:
        :return:
        """
        # temp file name, trim from video name
        # use absolute path to avoid confusion
        # srt_temp = os.path.splitext(os.path.basename(self.video_path))[0] + '.srt'
        srt_temp = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.video_path))[0] + '.srt')
        self.save_srt(subtitle, srt_temp)
        # merge soft subtitle, using ffmpeg library
        # output_file = '../output/'+ os.path.splitext(self.video_path)[0] + '.mp4'
        output_file = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.video_path))[0] + '_sub.mp4')
        video = ffmpeg.input(self.video_path)
        srt = ffmpeg.input(srt_temp)
        output = ffmpeg.output(video, srt, output_file, c='copy', scodec='mov_text')
        ffmpeg.run(output, overwrite_output=False)  # maybe overwrite input, caution
        logger.info(f'Merged soft subtitle to {output_file}')


async def main():
    for path in glob.glob("../input/*.*"):
        detector = VideoDetector(path)
        subtitle = detector.subtitle_detect()
        subtitle = await detector.subtitle_translate(subtitle)
        detector.merge_soft_sub(subtitle)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
