# Subtitle Translator

## 这个项目打算做什么？

1. 将视频中的硬字幕ocr提取，（paddleocr）
2. 翻译成设定的小语种，（google translate）
3. 将翻译好的字幕 以字幕轨道存入视频 或 硬压制到视频画面中。（use ffmpeg）

出于便于修改的目的，本项目使用 python 语言编写。


项目参考了 [Video-Subtitle-Extractor](https://github.com/YaoFANGUK/video-subtitle-extractor)