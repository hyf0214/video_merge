from datetime import datetime
import uvicorn
from pydantic import BaseModel
import requests
from fastapi import FastAPI, HTTPException
from tencentcloud.common import credential
from tencentcloud.vod.v20180717 import vod_client, models
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
import json
import os
from moviepy.editor import VideoFileClip

class VideoMerge(BaseModel):
    file_id: str
class Video(FastAPI):
    httpProfile = HttpProfile(endpoint="vod.tencentcloudapi.com")
    clientProfile = ClientProfile(httpProfile=httpProfile)
    def __init__(self):
        super().__init__()
        self.cred = credential.Credential("AKIDsrihIyjZOBsjimt8TsN8yvv1AMh5dB44", "CPZcxdk6W39Jd4cGY95wvupoyMd0YFqW")
        self.client_vod = vod_client.VodClient(self.cred, "", self.clientProfile)
        self.concurrent_upload_number = 5
        self.SubAppId = int(os.environ.get("SUBAPPID", 1324682537))
        self.merge_class_id = int(os.environ.get("MERGECLASSID", 	1181033))
        self.split_class_id = int(os.environ.get("SPLITCLASSID", 1182368))
        self.fade_duration = float(os.environ.get("FADETIME", 0.5))

    def get_vod_client(self, fileid: str) -> tuple[str, str, str]:
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
        print(data)
        # get_url = data["MediaInfoSet"][0]["BasicInfo"]["MediaUrl"]
        transcoding_url = data["MediaInfoSet"][0]["TranscodeInfo"]["TranscodeSet"][1]["Url"]
        print(transcoding_url)
        get_intranet_media_url = data["MediaInfoSet"][0]["BasicInfo"]["IntranetMediaUrl"]
        file_type = data["MediaInfoSet"][0]["BasicInfo"]["Type"]
        name = data["MediaInfoSet"][0]["BasicInfo"]["Name"]
        return get_intranet_media_url, file_type, name

    def download_file(self, url: str, seve_path: str) -> bool:
        try:
            response = requests.get(url, stream=True)
            with open(seve_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):  # 每次下载 8192 字节的数据
                    if chunk:
                        f.write(chunk)
            return True
        except TencentCloudSDKException as err:
            raise HTTPException(status_code=500, detail={
                "code": 500,
                "level": "RED",
                "message": "保存vod文件出错",
                "data": str(err)
            })

app = Video()
@app.post("/api/download_vod")
def download_vod(video_merge: VideoMerge):
    ap = Video()
    file_id = video_merge.file_id
    print(f"任务开始时间:{datetime.now()}")
    video_url, file_type, name = ap.get_vod_client(file_id)
    print(f"vod获取url结束：{datetime.now()}")
    down_video_path = f"/app/temp/{name}.{file_type}"
    # print(f"开始下载：{datetime.now()}")
    # ap.download_file(video_url, down_video_path)
    # print(f"下载结束:{datetime.now()}")
    return True

if __name__ == "__main__":

    # uvicorn.run(app, host="0.0.0.0", port=8888)

    # # 提取总秒数并计算毫秒
    # total_seconds = int(124144.87)
    #
    # milliseconds = td.microseconds
    # print(milliseconds)
    # # 计算小时、分钟和秒
    # hours, remainder = divmod(total_seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    #
    # # 格式化时间戳
    # timestamp = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    # print(timestamp)
    clip = VideoFileClip("D:/口感（3）_0_conding.mp4")
    print(clip.duration)
