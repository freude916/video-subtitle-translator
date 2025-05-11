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
                return "", 0.7 # 提醒可能存在未出现完全的字幕
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

    def subtitle_detect(self) -> List[Tuple[int, int, str, float]]:
        """
        核心逻辑1，执行字幕检测
        :return:
        """
        prev_frame_no = 0  # 扫完的帧
        current_frame_no = 0  # 正准备扫的帧
        prev_subtitle: str | None = None

        subtitle_result: List[Tuple[int, int, str, float]] = []

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
                    current_frame_no += 1 # 少跳一点
                    prev_subtitle = ""
                    continue
                current_subtitle: str = current_full_result[0]
                confidence: float = current_full_result[1]
                current_subtitle_begin = self.find_subtitle_boundary(prev_frame_no, current_frame_no, prev_subtitle)

                jump_end = self.find_subtitle_change(current_frame_no, current_subtitle)

                if current_frame_no > 100:
                    pass

                current_subtitle_end = self.find_subtitle_boundary(jump_end - self.frame_gap, jump_end, current_subtitle)

                subtitle_result.append((current_subtitle_begin, current_subtitle_end, current_subtitle, confidence))

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

    async def subtitle_translate(self, subtitle: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
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


subtitle = [(108, 115.5, '盗仓門', 0.5814619660377502), (116.5, 153.0, '盛沧澜', 0.9970487952232361),
            (154.0, 163.0, '瞧见了吗', 0.9976983070373535), (169.0, 196.5, '那护国公的千金', 0.9897699952125549),
            (197.5, 223.5, '竟追到了这里', 0.9917173385620117), (225.5, 259.5, '真是不害', 0.9998215436935425),
            (263.5, 290.5, '是我也不甘心', 0.9988496899604797),
            (298.5, 345.0, '满心期盼等候的心上', 0.9960343241691589),
            (346.0, 390.0, '人刚回来就另娶了她人', 0.9953144788742065), (426.0, 439.0, '小姐', 0.9998826384544373),
            (449.0, 471.5, '我们回去吧', 0.9997744560241699),
            (479.5, 504.5, '裴将军是不会见我们的', 0.9966943860054016),
            (516.5, 541.5, '裴将军是不会见我们的', 0.9777539968490601),
            (542.5, 562.5, '老爷正在找你', 0.9839210510253906), (572.5, 606.5, '若是知道你来了这', 0.9969412088394165),
            (616.5, 631.5, '恐怕', 0.99873948097229), (677.5, 680.0, '今', 0.7643079161643982),
            (681.0, 757.5, '日我只想听他亲口说一句', 0.9645965099334717),
            (785.5, 848.5, '他已经真心爱上了别人', 0.9244571924209595),
            (868.5, 913.0, '要退婚于我', 0.9957104921340942), (941.0, 943.5, '面', 0.9542027115821838),
            (997.5, 1036.0, '刀', 0.9594619870185852), (1084.0, 1119.0, '裴某今日大婚', 0.9972534775733948),
            (1133.0, 1179.5, '婚服尚未来得及更换', 0.9841243028640747),
            (1185.5, 1234.5, '并听闻盛小姐在此等候', 0.9978328943252563),
            (1252.5, 1281.0, '有失远迎了', 0.9979439973831177), (1313.0, 1334.0, '裴昭', 0.9949221014976501),
            (1356.0, 1395.0, '我只问你一次', 0.9983322620391846),
            (1403.0, 1455.5, '你是否真心退婚于我', 0.9664162397384644), (1459.5, 1462.0, '囍', 0.7538700699806213),
            (1488.0, 1512.5, '囍', 0.6021003723144531), (1514.5, 1563.5, '我已请得一道退婚圣旨', 0.992054283618927),
            (1575.5, 1607.0, '你是想抗旨不尊', 0.997049868106842), (1619.0, 1657.5, '被满门抄斩', 0.9710063934326172),
            (1671.5, 1730.5, '又或者盛姑娘愿意进来', 0.9946429133415222),
            (1736.5, 1774.0, '给我们作个床伴', 0.9973735213279724),
            (1798.0, 1841.0, '我倒也不介意', 0.9598219990730286), (1853.0, 1855.5, '囍', 0.5329992771148682),
            (1967.5, 1988.5, '裴昭', 0.9991567730903625), (2264.5, 2299.0, '那这算什么', 0.9997236132621765),
            (2501.0, 2556.5, '十年前', 0.995527446269989), (3062.5, 3096.5, '多谢小姐救命之恩', 0.9968143105506897),
            (3106.5, 3132.5, '你家在哪', 0.9996513724327087), (3133.5, 3172.0, '我让人送你回去', 0.9996028542518616),
            (3260.0, 3310.0, '家人都已死于战乱', 0.9982414245605469),
            (3360.0, 3391.0, '我已没有家了', 0.9987406730651855), (3401.0, 3422.5, '既如此', 0.9994029998779297),
            (3428.5, 3456.5, '那以后便跟着我吧', 0.9990864992141724), (3480.5, 3505.0, '你好好表现', 0.995815098285675),
            (3506.0, 3543.0, '日后若是获了军功', 0.9974322319030762),
            (3549.0, 3587.5, '你就可以衣食无忧了', 0.996820867061615),
            (3763.5, 3808.5, '家父临终有言', 0.9996432662010193), (3809.5, 3848.0, '即便死', 0.9987940788269043),
            (3854.0, 3898.5, '家传的玉簪也不能丢', 0.9539003372192383),
            (3920.5, 3951.5, '以后裴昭的命', 0.9984379410743713), (3971.5, 4006.5, '便是小姐的了', 0.9995988011360168),
            (4302.5, 4350.0, '这支玉簪算什么', 0.9845094680786133),
            (4370.0, 4427.5, '你的承诺算什么', 0.9995078444480896), (4435.5, 4456.0, '囍', 0.7254359126091003),
            (4498.0, 4574.0, '我盛沧澜对你而言又算什么', 0.9988043308258057),
            (4606.0, 4636.5, '自然是算你', 0.9174257516860962), (4656.5, 4698.5, '一厢情愿', 0.9927920699119568),
            (4762.5, 4801.0, '裴某心上人已有身孕', 0.9976029992103577),
            (4825.0, 4853.0, '不便被扰', 0.9991797804832458), (4879.0, 4881.5, '9', 0.5140502452850342),
            (4953.5, 4985.0, '请回吧', 0.9994271397590637), (5013.0, 5015.5, '孝', 0.6377304196357727),
            (5029.5, 5037.0, '裴昭新', 0.9756647944450378), (5038.0, 5088.5, '宋娇娇', 0.8678662776947021),
            (5089.5, 5103.5, '裴昭新婚妻子', 0.9576156139373779),
            (5104.5, 5130.0, '妾身与将军情投意合', 0.9975085854530334),
            (5134.0, 5169.0, '还望姑娘成全', 0.9987371563911438), (5193.0, 5225.0, '天下姻缘', 0.9727902412414551),
            (5231.0, 5264.0, '莫要强求为好', 0.9987258315086365),
            (5460.0, 5528.5, '你说过你的命是我的', 0.9987468719482422),
            (5560.5, 5622.0, '那你现在便还我', 0.9993332028388977), (5758.0, 5760.5, '勤', 0.6887478232383728)]


async def main():
    print(await detector.subtitle_translate(subtitle))


# asyncio.run(main())

if __name__ == '__main__':
    detector = VideoDetector('/home/zed/Documents/Codes/Python/Subtitle/input/test.mp4')
    subtitle = detector.subtitle_detect()
    print(subtitle)

    # detector.subtitle_translate(subtitle)

    # detector.merge_soft_sub(subtitle)
