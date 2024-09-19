import asyncio
import os
import json
import random
import base64
import time
import re
import uuid

import imageio
import requests
import logging

from typing import List, Dict
from datetime import timedelta

import uvicorn
import sentry_sdk
import webvtt
from sentry_sdk.integrations import aiohttp
from webvtt.models import Timestamp
from moviepy import video
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, ImageClip

from qcloud_vod.vod_upload_client import VodUploadClient
from qcloud_vod.model import VodUploadRequest

from fastapi import FastAPI, HTTPException
from fastapi_healthcheck import HealthCheckFactory, healthCheckRoute
from tencentcloud.common import credential
from tencentcloud.vod.v20180717 import vod_client, models
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException


class VideoMerge(BaseModel):
    split_file_id: str
    video_path: str
    details: List[Dict]

    class Config:
        extra = "allow"


RGX_TIMESTAMP_MAGNITUDE_DELIM = r"[,.:，．。：]"
RGX_TIMESTAMP_FIELD = r"[0-9]+"
RGX_TIMESTAMP_FIELD_OPTIONAL = r"[0-9]*"
RGX_TIMESTAMP_PARSEABLE = r"^{}$".format(
    "".join(
        [
            RGX_TIMESTAMP_MAGNITUDE_DELIM.join(["(" + RGX_TIMESTAMP_FIELD + ")"] * 3),
            RGX_TIMESTAMP_MAGNITUDE_DELIM,
            "?",
            "(",
            RGX_TIMESTAMP_FIELD_OPTIONAL,
            ")",
        ]
    )
)
TS_REGEX = re.compile(RGX_TIMESTAMP_PARSEABLE)


def timedelta_to_timestamp(delta: timedelta) -> str:
    SECONDS_IN_HOUR = 60 * 60 * 24
    HOURS_IN_DAY = 24 * SECONDS_IN_HOUR
    SECONDS_IN_MINUTE = 60
    hrs, secs_remainder = divmod(delta.seconds, SECONDS_IN_HOUR)
    hrs += delta.days * HOURS_IN_DAY
    mins, secs = divmod(secs_remainder, SECONDS_IN_MINUTE)
    msecs = delta.microseconds
    return "%02d:%02d:%02d.%03d" % (hrs, mins, secs, msecs)


def srt_timestamp_to_timedelta(timestamp: str) -> timedelta:
    match = TS_REGEX.match(timestamp)
    if match is None:
        raise Exception("Unparseable timestamp: {}".format(timestamp))
    hrs, mins, secs, msecs = [int(m) if m else 0 for m in match.groups()]
    return timedelta(hours=hrs, minutes=mins, seconds=secs, milliseconds=msecs)


class Video(FastAPI):
    httpProfile = HttpProfile(endpoint="vod.tencentcloudapi.com")
    clientProfile = ClientProfile(httpProfile=httpProfile)

    def __init__(self):
        super().__init__()
        self.logger = self.init_logger()
        self.cred = credential.Credential("AKIDsrihIyjZOBsjimt8TsN8yvv1AMh5dB44", "CPZcxdk6W39Jd4cGY95wvupoyMd0YFqW")
        self.client_vod = vod_client.VodClient(self.cred, "", self.clientProfile)
        self.client_down = VodUploadClient("AKIDCR4fQDyonkfUME8AKTVZZWK2kBXfhgfX", "vnjnsu14425FNYr5RsMpNyibcsEglwdV")
        self.concurrent_upload_number = 5
        self.SubAppId = int(os.environ.get("VOD_SUB_APP_ID", 1500032974))
        self.merge_class_id = int(os.environ.get("VOD_MERGE_MATERIAL", 1196658))
        self.split_class_id = int(os.environ.get("VOD_SPLIT_MATERIAL", 1196656))
        self.fade_duration = float(os.environ.get("FADETIME", 0.5))

    @staticmethod
    def get_log_level_from_env(env_key: str):
        log_level = os.environ.get(env_key, "DEBUG")
        log_level = log_level.upper()
        if log_level == "DEBUG":
            log_level = logging.DEBUG
        elif log_level == "WARNING":
            log_level = logging.WARNING
        elif log_level == "ERROR":
            log_level = logging.ERROR
        else:  # fallback to Info level
            log_level = logging.INFO
        return log_level

    @staticmethod
    def define_healthcheck_filter():
        class HealthCheckFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                # filter all GET /health HTTP/1.1" 200 OK
                return record.getMessage().find("/health") == -1 and record.getMessage().endswith("200 OK")

    def init_logger(self):
        logger = logging.getLogger("uvicorn")
        logger.handlers.clear()
        log_level = self.get_log_level_from_env("LOGGER_LOG_LEVEL")
        logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(filename)s#%(funcName)s:%(lineno)d] %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self.define_healthcheck_filter()
        return logger

    def get_vod_client(self, fileid: str) -> tuple[str, str, str]:
        try:
            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = models.DescribeMediaInfosRequest()
            params = {
                "SubAppId": self.SubAppId,
                "FileIds": [fileid],
                "ClassIds": [self.split_class_id]
            }
            req.from_json_string(json.dumps(params))
            # 返回的resp是一个DescribeMediaInfosResponse的实例，与请求对象对应
            resp = self.client_vod.DescribeMediaInfos(req)
            # 输出json格式的字符串回包
            data = json.loads(resp.to_json_string())
            self.logger.info(data)
            get_url = data["MediaInfoSet"][0]["BasicInfo"]["MediaUrl"]
            # get_IntranetMediaUrl = data["MediaInfoSet"][0]["BasicInfo"]["IntranetMediaUrl"]
            file_type = data["MediaInfoSet"][0]["BasicInfo"]["Type"]
            name = data["MediaInfoSet"][0]["BasicInfo"]["Name"]
            return get_url, file_type, name
        except TencentCloudSDKException as err:
            self.logger.error(f"获取文件信息失败:{err}")
            raise HTTPException(status_code=500, detail={
                "code": 500,
                "level": "RED",
                "message": "服务器获取vod失败",
                "data": str(err)
            })

    async def upload_vod(self, video_path: str) -> str:
        client = self.client_down
        request = VodUploadRequest()
        request.MediaFilePath = video_path
        request.ConcurrentUploadNumber = self.concurrent_upload_number
        request.ClassId = self.merge_class_id
        request.SubAppId = self.SubAppId
        try:
            response = client.upload("ap-shanghai", request)
            return response.FileId
        except Exception as e:
            self.logger.error(f"上传vod失败{e}")
            raise HTTPException(status_code=500, detail={
                "code": 500,
                "level": "RED",
                "message": "服务器上传vod失败",
                "data": str(e)
            })

    def vtt_to_base64(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                vtt_data = f.read()
                # 使用base64进行编码
                base64_data = base64.b64encode(vtt_data).decode('utf-8')
                return base64_data
        except FileNotFoundError:
            self.logger.error(f"vtt2b64 Error: 文件 '{file_path}' 未找到。")
        except Exception as e:
            self.logger.error(f"vtt2b64 Error: 发生异常 '{e}'.")

    def image_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                # 读取图片文件内容
                image_data = image_file.read()
                # 使用base64进行编码
                base64_data = base64.b64encode(image_data)
                # 将字节序列转换为字符串
                base64_string = base64_data.decode('utf-8')
                return base64_string
        except FileNotFoundError:
            self.logger.error(f"img2b64 Error: 文件 '{image_path}' 未找到。")
        except Exception as e:
            self.logger.error(f"img2b64 Error: 发生异常 '{e}'.")

    async def vod_wallpaper_subtitle(self, file_id: str, image_path: str, vtt_path: str) -> bool:
        vtt_base64 = self.vtt_to_base64(vtt_path)
        img_base64 = self.image_to_base64(image_path)
        try:
            req = models.ModifyMediaInfoRequest()
            params = {
                "SubAppId": self.SubAppId,
                "ClassIds": [self.split_class_id],
                "FileId": file_id,
                "CoverData": img_base64,
                "AddSubtitles": [
                    {
                        "Language": "cn",
                        "Format": "vtt",
                        "Content": vtt_base64
                    }
                ]
            }
            req.from_json_string(json.dumps(params))

            # 返回的resp是一个ModifyMediaInfoResponse的实例，与请求对象对应
            resp = self.client_vod.ModifyMediaInfo(req)
            return True
        except TencentCloudSDKException as err:
            self.logger.error(f"绑定视频字幕、壁纸信息失败 ：{err}")

    @staticmethod
    def create_count(count_path: str, count: int) -> bool:
        with open(count_path, "w", encoding="utf-8") as f:
            f.write(str(count))
        return True

    async def download_vod(self, file_id: str, video_path: str, user_count: int):
        video_url1, file_type, name = self.get_vod_client(file_id)
        video_url = video_url1 + f"?download_name={name}.{file_type}"
        prefix = video_path.split("split")[0] + "split"
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        down_video_path = f"{prefix}/{name}.{file_type}"
        self.logger.info("开始下载视频：:%s", down_video_path)
        self.download_file(video_url, down_video_path)
        down_count_path = f"{prefix}/{name}.metadata"
        self.create_count(down_count_path, user_count)
        return down_video_path, name, down_count_path

    def download_file(self, url: str, seve_path: str) -> bool:
        try:
            response = requests.get(url, stream=True)
            with open(seve_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):  # 每次下载 8192 字节的数据
                    if chunk:
                        f.write(chunk)
            return True
        except TencentCloudSDKException as err:
            self.logger.error(f"保存vod文件出错{err}")
            raise HTTPException(status_code=500, detail={
                "code": 500,
                "level": "RED",
                "message": "保存vod文件出错",
                "data": str(err)
            })

    async def group_consecutive(self, nums):
        result = []
        i = 0
        while i < len(nums):
            if i == len(nums) - 1 or int(nums[i + 1]["index"]) != int(nums[i]["index"]) + 1:
                result.append(nums[i])
            else:
                sublist = [nums[i]]
                while i < len(nums) - 1 and int(nums[i + 1]["index"]) == int(nums[i]["index"]) + 1:
                    sublist.append(nums[i + 1])
                    i += 1
                result.append(sublist)
            i += 1
        self.logger.info(f"group_data:{result}")
        return result

    def time_to_seconds(self, time_str) -> float:
        # 分离时间部分和毫秒部分
        time_part, ms = time_str.split('.')
        # 分离小时、分钟和秒
        h, m, s = map(int, time_part.split(':'))
        # 计算总秒数
        total_seconds = h * 3600 + m * 60 + s + int(ms) / 1000
        return total_seconds

    def time_split_video(self, video_info, video_clip, index, results_num):
        time_data = video_info["time_point"]  # "00:03:10.400 --> 00:03:11.366"
        start_time = self.time_to_seconds(time_data.split(" ")[0])
        oly_end_time = self.time_to_seconds(time_data.split(" ")[2])
        if index != results_num:
            end_time = oly_end_time + self.fade_duration
            self.logger.info(f"截取视频 start_time:{start_time}, end_time:{end_time}")
            video_clip1 = self.extend_footage(video_clip, end_time, oly_end_time)
            subclip = video_clip1.subclip(start_time, end_time)
        else:
            self.logger.info(f"截取视频 start_time:{start_time}, end_time:{oly_end_time}")
            video_clip1 = self.extend_footage(video_clip, oly_end_time)
            subclip = video_clip1.subclip(start_time, oly_end_time)
        return subclip

    @staticmethod
    def reade_count_delete(video_path: str, count_path: str) -> bool:
        with open(count_path, 'r') as file:
            number_str = file.read().strip()  # strip() 去除可能存在的空白字符
            # 将字符串转换为整数
            number = int(number_str) - 1
        if number == 0:
            os.remove(video_path)
            os.remove(count_path)
        else:
            with open(count_path, 'w') as file:
                # 将新的整数转换回字符串并写入文件
                file.write(str(number))
        return True

    @staticmethod
    def convert_seconds_to_timestamp(seconds):
        print(seconds)
        td = timedelta(seconds=seconds)
        time_data = Timestamp.from_string(timedelta_to_timestamp(td))  # 00:00:00.000
        print(time_data)
        return time_data


    async def video_deduplication(self, video_clip: VideoFileClip, background: ColorClip, index: int,
                                  clip_list_path: str) -> bool:
        random_float = round(random.uniform(1, 1.1), 2)
        new_video_w = video_clip.w * random_float
        new_video_h = video_clip.h * random_float
        new_video_h = ((new_video_h // 16) + 1) * 16 if new_video_h % 16 != 0 else new_video_h
        # 放大比例
        resized_clip = video_clip.resize((new_video_w, new_video_h))
        self.logger.info(
            f"原素材尺寸：{video_clip.w}*{video_clip.h},放大倍数：{random_float}，新视频尺寸：{new_video_w}*{new_video_h}")
        # 重叠素材
        overwrite_footage = VideoFileClip("/app/footage.mp4")
        # 设置透明度
        overwrite_footage = overwrite_footage.set_opacity(0.03)

        x_cover_position = random.randint(-1000, 0)
        y_cover_position = random.randint(0, 50)

        try:
            final_clip = CompositeVideoClip([resized_clip,
                                             overwrite_footage.set_duration(resized_clip.duration).set_position(
                                                 (x_cover_position, y_cover_position))], size=resized_clip.size)

            # 移动位置
            cha_w = int((new_video_w - 1080) / 2)
            cha_h = int((new_video_h - 1920) / 2)

            y_position = random.randint(-abs(cha_h), 0)
            x_position = random.randint(-abs(cha_w), 0)

            video_clip_resized = CompositeVideoClip(
                [background.set_duration(resized_clip.duration), final_clip.set_position((x_position, y_position))])
            video_clip_resized.write_videofile(f"{clip_list_path}/{index}.mp4", codec='libx264', threads=4)
            video_clip_resized.close()
            overwrite_footage.close()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail={f"视频去重失败失败:{e}"})


    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


    def video_crossfadein(self, video_clips):
        clips = []
        start_time = 0
        for i, video1 in enumerate(video_clips):
            if i == 0:
                clips.append(video1)
                start_time += video1.duration
            else:
                # 将视频按顺序合并，并设置淡入效果
                clips.append(video1.set_start(start_time - self.fade_duration).crossfadein(self.fade_duration))
                start_time += video1.duration - self.fade_duration
        # 创建最终合成的视频
        final_video = CompositeVideoClip(clips)
        return final_video, clips


    def extend_footage(self, video_clip: VideoFileClip, end_time, oly_time=0.0):
        time_num = video_clip.duration - end_time  # 总时长-加淡化的时间 小于0 定帧
        oly_num = video_clip.duration - oly_time  # 总时间-不加淡化的时间 大于0 去声音
        if oly_time != 0.0 and time_num < 0:
            # 素材不够获取视频的最后一帧 (不是最后一个+素材不够）
            frame = video_clip.get_frame(video_clip.duration - 0.5)
            last_frame_clip = ImageClip(frame).set_duration(-time_num)
            self.logger.info(f"素材不够获取视频的最后一帧 定格：{-time_num}")
            if oly_num > 0:
                # 定帧 + 多余原素材 去掉多余原素材音频
                self.logger.info("去除原视频多余音频")
                part1 = video_clip.subclip(0, oly_time)
                part2 = video_clip.subclip(oly_time, oly_time + oly_num).without_audio()  # 静音部分
                extended_clip = concatenate_videoclips([part1, part2, last_frame_clip])
            else:
                extended_clip = concatenate_videoclips([video_clip, last_frame_clip])
        elif oly_time != 0.0 and time_num >= 0:
            # 不定帧 结束时间在素材时间内，去除淡化音频 （不是最后一个+素材够）
            part1 = video_clip.subclip(0, oly_time)
            part2 = video_clip.subclip(oly_time, end_time).without_audio()
            part3 = video_clip.subclip(end_time, video_clip.duration)
            extended_clip = concatenate_videoclips([part1, part2, part3])
            # 如果视频已经足够长，返回原视频
        elif oly_time == 0.0 and time_num < 0:
            frame = video_clip.get_frame(video_clip.duration - 0.5)
            last_frame_clip = ImageClip(frame).set_duration(-time_num)
            extended_clip = concatenate_videoclips([video_clip, last_frame_clip])
        else:
            extended_clip = video_clip
        return extended_clip


    async def video_merge(self, details: list, video_path: str, image_path: str, merge_video_path: str,
                          clip_list_path: str) -> bool:
        results = await self.group_consecutive(details)
        background = ColorClip((1080, 1920), color=(0, 0, 0))
        video_clips = []
        merge_clip_list = None
        clips = None
        with VideoFileClip(video_path) as video_clip:
            indexs = 0
            results_num = len(results)
            for index, result in enumerate(results, 1):
                # 连续index合并一起截取
                if isinstance(result, list):
                    time_list = []
                    indexs += 1
                    for video_info in result:
                        time_data = video_info["time_point"]  # "00:03:10.400 --> 00:03:11.366"
                        start_time = self.time_to_seconds(time_data.split(" ")[0])
                        end_time = self.time_to_seconds(time_data.split(" ")[2])
                        time_list.append(start_time)
                        time_list.append(end_time)
                    start_time = time_list[0]
                    end_time = time_list[-1]
                    # 不是合成的最后一个 多截取淡出时间
                    if results_num != index:
                        oly_time = time_list[-1]
                        end_time = time_list[-1] + self.fade_duration
                        extended_clip = self.extend_footage(video_clip, end_time, oly_time)
                    else:
                        extended_clip = self.extend_footage(video_clip, end_time)
                    subclip_oly = extended_clip.subclip(start_time, end_time)
                else:
                    indexs += 1
                    subclip_oly = self.time_split_video(result, video_clip, index, results_num)
                await self.video_deduplication(subclip_oly, background, indexs, clip_list_path)

        try:
            files = sorted(os.listdir(clip_list_path), key=self.natural_sort_key)
            for file in files:
                if file.endswith('.mp4'):  # 假设视频文件为 mp4 格式
                    video_clip = VideoFileClip(os.path.join(clip_list_path, file))
                    video_clips.append(video_clip)
            merge_clip_list, clips = self.video_crossfadein(video_clips)
            # with concatenate_videoclips(clips, method="compose") as merge_clip_list:
            # 获取视频壁纸 提取第一帧图像
            frame = merge_clip_list.get_frame(0)  # 获取第一帧
            # 保存第一帧为图片
            imageio.imwrite(image_path, frame)
            merge_clip_list.write_videofile(merge_video_path, threads=4)
            return True
        except Exception as err:
            raise HTTPException(status_code=500, detail={f"合并视频失败:{err}"})
        finally:
            if clips:
                for clip in clips:
                    clip.close()
            # 关闭所有子片段
            if merge_clip_list:
                merge_clip_list.close()
            for clip in video_clips:
                if clip:
                    clip.close()


    async def regroup_vtt(self, details: list, vtt_path: str) -> bool:
        start = []
        miao = []
        for index, detail in enumerate(details):
            time_end_list = []
            text = detail["content"]
            time_point = detail["time_point"]
            a = self.time_to_seconds(time_point.split(" ")[0])
            b = self.time_to_seconds(time_point.split(" ")[2])
            num = b - a
            if index == 0:
                star_time = "00:00:00.000"
            else:
                star_time = start[index - 1]
                num = miao[index - 1] + num
            end_time = self.convert_seconds_to_timestamp(num)
            start.append(end_time)
            miao.append(self.time_to_seconds(end_time))
            time_end_list.append(index + 1)
            time_end_list.append(star_time)
            time_end_list.append(end_time)
            time_end_list.append(text)
            with open(vtt_path, "a", encoding="utf-8") as file:
                file.write(f"{time_end_list[0]}\n")
                file.write(f"{time_end_list[1]} --> {time_end_list[2]}\n")
                file.write(f"{time_end_list[3]}\n")
                file.write("\n")
        return True


    async def new_regroup_vtt(self, details: list, vtt_path: str) -> str:
        start = []
        miao = []
        details = await self.group_consecutive(details)
        index = -1
        for detail in details:
            if isinstance(detail, list):
                index += 1
                item_end_list = []
                for i, item in enumerate(detail):
                    if i == 0:
                        text = item["content"]
                        time_point = item["time_point"]
                        a = self.time_to_seconds(time_point.split(" ")[0])
                        b = self.time_to_seconds(time_point.split(" ")[2])
                        num = b - a
                        if start:
                            num += miao[-1]
                        start_time = self.convert_seconds_to_timestamp(miao[-1]) if start else "00:00:00.000"
                        end_time = self.convert_seconds_to_timestamp(num)
                        start.append(end_time)
                        miao.append(num)
                        item_end_list.append(b)
                        # captions.append(webvtt.Caption(text, start_time, end_time, identifier=str(index + 1)))
                        with open(vtt_path, "a", encoding="utf-8") as file:
                            file.write(f"{index + 1}\n")
                            file.write(f"{start_time} --> {end_time}\n")
                            file.write(f"{text}\n")
                            file.write("\n")
                    else:
                        index += 1
                        text = item["content"]
                        time_point = item["time_point"]
                        a = self.time_to_seconds(time_point.split(" ")[0])
                        b = self.time_to_seconds(time_point.split(" ")[2])
                        num = b - a
                        if start:
                            num += miao[-1]
                        s = miao[-1] + (a - item_end_list[-1])
                        e = num + (a - item_end_list[-1])
                        start_time = self.convert_seconds_to_timestamp(s) if start else "00:00:00.000"
                        end_time = self.convert_seconds_to_timestamp(e)
                        start.append(end_time)
                        miao.append(e)
                        # captions.append(webvtt.Caption(text, start_time, end_time, identifier=str(index+1)))
                        item_end_list.append(b)
                        with open(vtt_path, "a", encoding="utf-8") as file:
                            file.write(f"{index + 1}\n")
                            file.write(f"{start_time} --> {end_time}\n")
                            file.write(f"{text}\n")
                            file.write("\n")
            else:
                index += 1
                text = detail["content"]
                time_point = detail["time_point"]
                a = self.time_to_seconds(time_point.split(" ")[0])
                b = self.time_to_seconds(time_point.split(" ")[2])
                num = b - a
                if start:
                    num += miao[-1]
                start_time = self.convert_seconds_to_timestamp(miao[-1]) if start else "00:00:00.000"
                end_time = self.convert_seconds_to_timestamp(num)
                start.append(end_time)
                miao.append(num)
                # captions.append(webvtt.Caption(text, start_time, end_time, identifier=str(index + 1)))
                with open(vtt_path, "a", encoding="utf-8") as file:
                    file.write(f"{index + 1}\n")
                    file.write(f"{start_time} --> {end_time}\n")
                    file.write(f"{text}\n")
                    file.write("\n")

        return vtt_path


    async def delete_folder(self, folder_path: str, retries=3, delay=5):
        """
        尝试删除文件夹及其内容，失败时重试指定次数。
        :param folder_path: 要删除的文件夹路径
        :param retries: 失败时重试的次数
        :param delay: 重试前的等待时间（秒）
        """
        for attempt in range(retries):
            try:
                if os.path.exists(folder_path):
                    # 逐个文件删除
                    for root, dirs, files in os.walk(folder_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                    # 删除空文件夹
                    os.rmdir(folder_path)
                    self.logger.info(f"文件夹{folder_path}删除成功！", )
                    return
            except Exception as e:
                self.logger.error(f"删除文件夹 '{folder_path}' 失败（尝试 {attempt + 1}/{retries}）：{e}")
                time.sleep(delay)


sentry_sdk.init(
    dsn="http://07fec9ca053b527cff012d0bea60b159@122.51.245.126:80/10",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
)

app = Video()
_healthChecks = HealthCheckFactory()
app.add_api_route('/health', endpoint=healthCheckRoute(factory=_healthChecks))


def generate_uuid():
    # 生成一个UUID
    unique_id = uuid.uuid4()
    # 从UUID中提取前8个字符
    folder_name = unique_id.hex[:8]
    return folder_name


@app.post("/api/record_video_merge")
async def record_video_merge(video_merge: VideoMerge):
    file_id = str(video_merge.split_file_id)
    details = video_merge.details
    video_path = video_merge.video_path
    user_count = int(video_merge.metadata.get("batch_count"))

    app.logger.info(f"视频合成任务开始:VideoMerge {video_merge}")
    cfs_video_path = f"/app/temp{video_path}"
    merge_path = f"/app/temp/{generate_uuid()}"
    clip_list_path = f"{merge_path}/clip_list"
    os.makedirs(clip_list_path, exist_ok=True)
    try:
        down_count_path = cfs_video_path.replace("_conding", "").replace(".mp4", ".metadata")
        if not os.path.exists(cfs_video_path):
            cfs_video_path, name, down_count_path = await app.download_vod(file_id, cfs_video_path, user_count)
            await app.download_vod(file_id, cfs_video_path, user_count)
        name = video_path.split("/")[-1].split(".")[0]
        merge_video_path = f"{merge_path}/{name}_merge.mp4"
        image_path = f"{merge_path}/{name}.png"

        await app.video_merge(details, cfs_video_path, image_path, merge_video_path, clip_list_path)
        app.logger.info(f"保存合并视频 url:{merge_video_path}")

        vtt_path = f"{merge_path}/{name}.vtt"
        await app.new_regroup_vtt(details, vtt_path)

        # 上传vod
        merge_fileid = await app.upload_vod(merge_video_path)
        app.logger.info(f"上传vod成功 fileID：{merge_fileid}")

        # 绑定壁纸与字幕
        await app.vod_wallpaper_subtitle(merge_fileid, image_path, vtt_path)
        vtt2base64 = app.vtt_to_base64(vtt_path)
        app.reade_count_delete(cfs_video_path, down_count_path)
        data = video_merge.model_dump()
        data.update(merge_file_id=merge_fileid,
                    merge_vtt_content=vtt2base64)
        return data
    except Exception as e:
        app.logger.error(e)
        raise
    finally:
        await app.delete_folder(merge_path)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
