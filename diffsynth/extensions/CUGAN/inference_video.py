import threading, sys, torch, os
from random import uniform
from multiprocessing import Queue
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip
from .upcunet_v3 import RealWaifuUpScaler
from time import time as ttime, sleep
from PIL import Image


class UpScalerMT(threading.Thread):
    def __init__(self, inp_q, res_q, model, p_sleep, nt, tile, cache, alpha):
        threading.Thread.__init__(self)
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.nt = nt
        self.p_sleep = p_sleep
        self.tile = tile
        self.cache_mode = cache
        self.alpha = alpha

    def inference(self, tmp):
        idx, np_frame = tmp
        with torch.no_grad():
            res = self.model(np_frame, self.tile, self.cache_mode, self.alpha)
        if self.nt > 1:
            sleep(uniform(self.p_sleep[0], self.p_sleep[1]))
        return (idx, res)

    def run(self):
        while 1:
            tmp = self.inp_q.get()
            if tmp == None:
                break
            self.res_q.put(self.inference(tmp))


class VideoRealWaifuUpScaler(object):
    def __init__(
        self, scale, weight_path, half=True, device="cuda", tile=0, cache_mode=1, alpha=1,
        nt=2, n_gpu=1, p_sleep=(0.005, 0.012), decode_sleep=0.002, encode_params=["-crf", "21", "-preset", "medium"]
    ):
        self.scale = scale
        self.weigth_path = weight_path
        self.half = half
        self.device = device
        self.tile = tile
        self.cache_mode = cache_mode
        self.alpha = alpha
        self.nt = nt
        self.n_gpu = n_gpu
        self.p_sleep = p_sleep
        self.decode_sleep = decode_sleep
        self.encode_params = encode_params

        if device == "cpu":
            print("video super resolution not support CPU mode")
            sys.exit(0)

    def start(self, repeated=False):
        if not repeated:
            self.inp_q = Queue(self.nt * self.n_gpu * 2)  # 抽帧缓存上限帧数
            self.res_q = Queue(self.nt * self.n_gpu * 2)  # 超分帧结果缓存上限

            for i in range(self.n_gpu):
                device = self.device + ":%s" % i
                model = RealWaifuUpScaler(self.scale, self.weigth_path, self.half, device)
                for _ in range(self.nt):
                    upscaler = UpScalerMT(
                        self.inp_q, self.res_q, model, self.p_sleep, self.nt, self.tile, self.cache_mode, self.alpha
                    )
                    upscaler.start()

    def process_image(self, image):
        t0 = ttime()
        model = RealWaifuUpScaler(self.scale, self.weigth_path, self.half, self.device)
        res = model(image, self.tile, self.cache_mode, self.alpha)
        res = Image.fromarray(res)
        t1 = ttime()
        print("Super Resolution Done. time cost: %.3f" % (t1 - t0))
        return res

    def __call__(self, inp_path, opt_path, out_name="output_super"):
        suffix = inp_path.split(".")[-1]
        os.makedirs(opt_path, exist_ok=True)
        output = os.path.join(opt_path, f"{out_name}.{suffix}")
        objVideoreader = VideoFileClip(filename=inp_path)
        w, h = objVideoreader.reader.size
        fps = objVideoreader.reader.fps
        total_frame = objVideoreader.reader.nframes
        if_audio = objVideoreader.audio

        if if_audio:
            tmp_audio_path = os.path.join(opt_path, "audio.m4a")
            objVideoreader.audio.write_audiofile(tmp_audio_path, codec="aac")
            writer = FFMPEG_VideoWriter(
                output,
                (w * self.scale, h * self.scale),
                fps,
                ffmpeg_params=self.encode_params,
                audiofile=tmp_audio_path,
            )
        else:
            writer = FFMPEG_VideoWriter(
                output,
                (w * self.scale, h * self.scale),
                fps,
                ffmpeg_params=self.encode_params,
            )

        now_idx = 0
        idx2res = {}
        t0 = ttime()
        for idx, frame in enumerate(objVideoreader.iter_frames()):
            self.inp_q.put((idx, frame))
            sleep(self.decode_sleep)  # 否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
            while 1:
                if self.res_q.empty():
                    break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            while 1:
                if now_idx not in idx2res:
                    break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1

        idx += 1
        while 1:
            while 1:
                if self.res_q.empty():
                    break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            while 1:
                if now_idx not in idx2res:
                    break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
            if self.inp_q.qsize() == 0 and self.res_q.qsize() == 0 and idx == now_idx:
                break
            sleep(0.02)

        for _ in range(self.nt * self.n_gpu):  # 全部结果拿到后，关掉模型线程
            self.inp_q.put(None)
        writer.close()

        if if_audio:
            os.remove(tmp_audio_path)
        t1 = ttime()
        print(f"Super Resolution Done, {os.path.basename(out_name)} time cost: {t1 - t0}")


if __name__ == "__main__":
    weight_paths = [
        (
            "../../../../../models/CUGAN/pro-denoise3x-up2x.pth",
            2,
        ),
        (
            "../../../../../models/CUGAN/pro-denoise3x-up3x.pth",
            3,
        ),
    ]

    video_input = "../../../../../data/breakdance.mp4"
    base_name = os.path.basename(video_input).split(".")[0]

    for weight_path, scale in weight_paths:
        for tile_mode in [0, 1, 5]:
            for cache_mode in [0, 1]:
                for alpha in [0.75, 1, 1.3]:
                    output_name = f"{base_name}_{scale}x_tile{tile_mode}_cache{cache_mode}_alpha{alpha}"
                    video_upsampler = VideoRealWaifuUpScaler(
                        scale, weight_path, tile=tile_mode, cache_mode=cache_mode, alpha=alpha
                    )
                    video_upsampler.start()
                    video_upsampler(video_input, "output", output_name)