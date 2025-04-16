import numpy as np
import argparse
import datetime
import torch
import cv2
import os
import moviepy.video.io.ImageSequenceClip

from dwpose import DWposeDetector, draw_pose
from util import size_calculate, warpAffine_kps

"""
    Detect dwpose from img, then align it by scale parameters
    img: frame from the pose video
    detector: DWpose
    scales: scale parameters
"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detect_resolution", type=int, default=768, help="detect_resolution"
    )
    parser.add_argument(
        "--image_resolution", type=int, default=768, help="image_resolution"
    )

    parser.add_argument(
        "--align_frame",
        type=int,
        default=0,
        help="the frame index of the video to align",
    )
    parser.add_argument(
        "--max_frame",
        type=int,
        default=300,
        help="maximum frame number of the video to align",
    )

    parser.add_argument("--imgfn", type=str, required=True, help="refer image path")
    parser.add_argument("--vidfn", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--outfn",
        type=str,
        default=None,
        help="output dir of the alignment visualization",
    )

    parser.add_argument("--end2end", action="store_true", help="use end2end model")
    parser.add_argument("--hand", action="store_true", help="add hand keypoints")
    parser.add_argument("--face", action="store_true", help="add face keypoints")

    return parser


def align_img(img, pose_ori, scales):
    body_pose = pose_ori["bodies"]["candidate"].copy()
    hands = pose_ori["hands"].copy()
    faces = pose_ori["faces"].copy()

    """
    计算逻辑:
    0. 该函数内进行绝对变换，始终保持人体中心点 body_pose[1] 不变
    1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
    2. 用点在图中的实际坐标来计算。
    3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
    4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
    注意：dwpose 输出是 (w, h)
    """

    # h不变，w缩放到原比例
    H_in, W_in, C_in = img.shape
    video_ratio = W_in / H_in
    body_pose[:, 0] = body_pose[:, 0] * video_ratio
    hands[:, :, 0] = hands[:, :, 0] * video_ratio
    faces[:, :, 0] = faces[:, :, 0] * video_ratio

    # scales of 10 body parts
    scale_neck = scales["scale_neck"]
    scale_face = scales["scale_face"]
    scale_shoulder = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand = scales["scale_hand"]
    scale_body_len = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_sum = 0
    count = 0
    scale_list = [
        scale_neck,
        scale_face,
        scale_shoulder,
        scale_arm_upper,
        scale_arm_lower,
        scale_hand,
        scale_body_len,
        scale_leg_upper,
        scale_leg_lower,
    ]
    for i in range(len(scale_list)):
        if not np.isinf(scale_list[i]):
            scale_sum = scale_sum + scale_list[i]
            count = count + 1
    for i in range(len(scale_list)):
        if np.isinf(scale_list[i]):
            scale_list[i] = scale_sum / count

    # offsets of each part
    offset = dict()
    offset["14_15_16_17_to_0"] = body_pose[[14, 15, 16, 17], :] - body_pose[[0], :]
    offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :]
    offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :]
    offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :]
    offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :]
    offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :]
    offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :]
    offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :]
    offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :]
    offset["hand_left_to_4"] = hands[1, :, :] - body_pose[[4], :]
    offset["hand_right_to_7"] = hands[0, :, :] - body_pose[[7], :]

    # neck
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_neck)

    neck = body_pose[[0], :]
    neck = warpAffine_kps(neck, M)
    body_pose[[0], :] = neck

    # body_pose_up_shoulder
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_face)

    body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[14, 15, 16, 17], :] = body_pose_up_shoulder

    # face
    # body_pose (14-left eye, 15-right eye, 0-neck)
    faces = face_align(faces, body_pose[[14, 15, 0], :], M)

    # shoulder
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_shoulder)

    body_pose_shoulder = body_pose[[2, 5], :]
    body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M)
    body_pose[[2, 5], :] = body_pose_shoulder

    # arm upper left
    c_ = body_pose[2]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_arm_upper)

    elbow = offset["3_to_2"] + body_pose[[2], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[3], :] = elbow

    # arm lower left
    c_ = body_pose[3]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_arm_lower)

    wrist = offset["4_to_3"] + body_pose[[3], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[4], :] = wrist

    # hand left
    c_ = body_pose[4]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_hand)

    hand = offset["hand_left_to_4"] + body_pose[[4], :]
    hand = warpAffine_kps(hand, M)
    hands[1, :, :] = hand

    # arm upper right
    c_ = body_pose[5]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_arm_upper)

    elbow = offset["6_to_5"] + body_pose[[5], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[6], :] = elbow

    # arm lower right
    c_ = body_pose[6]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_arm_lower)

    wrist = offset["7_to_6"] + body_pose[[6], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[7], :] = wrist

    # hand right
    c_ = body_pose[7]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_hand)

    hand = offset["hand_right_to_7"] + body_pose[[7], :]
    hand = warpAffine_kps(hand, M)
    hands[0, :, :] = hand

    # body len
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_body_len)

    body_len = body_pose[[8, 11], :]
    body_len = warpAffine_kps(body_len, M)
    body_pose[[8, 11], :] = body_len

    # leg upper left
    c_ = body_pose[8]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_leg_upper)

    knee = offset["9_to_8"] + body_pose[[8], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[9], :] = knee

    # leg lower left
    c_ = body_pose[9]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_leg_lower)

    ankle = offset["10_to_9"] + body_pose[[9], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[10], :] = ankle

    # leg upper right
    c_ = body_pose[11]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_leg_upper)

    knee = offset["12_to_11"] + body_pose[[11], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[12], :] = knee

    # leg lower right
    c_ = body_pose[12]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx, cy), 0, scale_leg_lower)

    ankle = offset["13_to_12"] + body_pose[[12], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[13], :] = ankle

    # none part
    body_pose_none = pose_ori["bodies"]["candidate"] == -1.0
    hands_none = pose_ori["hands"] == -1.0
    faces_none = pose_ori["faces"] == -1.0

    body_pose[body_pose_none] = -1.0
    hands[hands_none] = -1.0
    if len(hands[np.isnan(hands)]) > 0:
        print("nan")
    faces[faces_none] = -1.0

    # last check nan -> -1.
    body_pose = np.nan_to_num(body_pose, nan=-1.0)
    hands = np.nan_to_num(hands, nan=-1.0)
    faces = np.nan_to_num(faces, nan=-1.0)

    # return
    pose_align = pose_ori.copy()
    pose_align["bodies"]["candidate"] = body_pose
    pose_align["hands"] = hands
    pose_align["faces"] = faces

    return pose_align


def face_align(faces, bodys, matrix):
    """
    参数：
        faces : numpy数组, 形状为 (n, 68, 2)
            n张人脸，每张人脸68个关键点，不可见点的值为-1
        bodys : numpy数组, 形状为 (n, k, 2)
            对应的身体参考关键点，k为参考点数量（需与对齐逻辑匹配）
    返回：
        face_aligned : numpy数组, 形状同faces
            对齐后的面部关键点，不可见点仍为-1
    """
    face_aligned = np.copy(faces)

    for i in range(faces.shape[0]):
        # --- 步骤1：选择用于对齐的源点（面部）和目标点（身体） ---
        src_points = []
        dst_points = []

        # 左眼中心（关键点36和39的平均）
        if (
            faces[i, 36, 0] != -1
            and faces[i, 36, 1] != -1
            and faces[i, 39, 0] != -1
            and faces[i, 39, 1] != -1
        ):
            left_eye_center = (faces[i, 36] + faces[i, 39]) / 2
            src_points.append(left_eye_center)
            dst_points.append(bodys[0])

        # 右眼中心（关键点42和45的平均）
        if (
            faces[i, 42, 0] != -1
            and faces[i, 42, 1] != -1
            and faces[i, 45, 0] != -1
            and faces[i, 45, 1] != -1
        ):
            right_eye_center = (faces[i, 42] + faces[i, 45]) / 2
            src_points.append(right_eye_center)
            dst_points.append(bodys[1])

        # 鼻子（关键点30）
        if faces[i, 30, 0] != -1 and faces[i, 30, 1] != -1:
            src_points.append(faces[i, 30])
            dst_points.append(bodys[2])

        # --- 步骤2：鲁棒性计算变换矩阵 ---
        if len(src_points) >= 2:  # 至少需要2个点计算仿射变换
            src_pts = np.array(src_points, dtype=np.float32)
            dst_pts = np.array(dst_points, dtype=np.float32)

            # 使用RANSAC过滤异常点
            transform_matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            if transform_matrix is not None:
                # --- 步骤3：仅对可见点应用变换 ---
                # 生成可见点掩码（排除-1值）
                visible_mask = np.logical_and(
                    faces[i, :, 0] != -1, faces[i, :, 1] != -1
                )
                visible_indices = np.where(visible_mask)[0]

                if len(visible_indices) > 0:
                    visible_points = faces[i, visible_indices, :].astype(np.float32)
                    aligned_points = cv2.transform(
                        visible_points.reshape(-1, 1, 2), transform_matrix
                    ).reshape(-1, 2)
                    face_aligned[i, visible_indices, :] = aligned_points
        else:
            # 变换矩阵计算失败，直接进行尺度变换
            face_aligned = warpAffine_kps(faces, matrix)

    return face_aligned


def run_align_video_with_filterPose_translate_smooth(args, demo_out, pose_out):
    vidfn = args.vidfn
    imgfn = args.imgfn

    video = cv2.VideoCapture(vidfn)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)

    print("height:", height)
    print("width:", width)
    print("fps:", fps)

    H_in, W_in = height, width
    H_out, W_out = size_calculate(H_in, W_in, args.detect_resolution)
    H_out, W_out = size_calculate(H_out, W_out, args.image_resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = DWposeDetector(device=device, keypoints_only=False, end2end=args.end2end)

    refer_img = cv2.imread(imgfn)
    output_refer, pose_refer = detector(
        refer_img,
        detect_resolution=args.detect_resolution,
        image_resolution=args.image_resolution,
        output_type="cv2",
        return_pose_dict=True,
    )
    body_ref_img = pose_refer["bodies"]["candidate"]
    hands_ref_img = pose_refer["hands"]
    faces_ref_img = pose_refer["faces"]
    subset_ref_img = pose_refer["bodies"]["subset"][0]
    output_refer = cv2.cvtColor(output_refer, cv2.COLOR_RGB2BGR)

    skip_frames = args.align_frame
    max_frame = args.max_frame
    pose_list, video_frame_buffer, video_pose_buffer = [], [], []

    for i in range(max_frame):
        ret, img = video.read()
        if img is None:
            break
        else:
            if i < skip_frames:
                continue
            video_frame_buffer.append(img)

        # estimate scale parameters by the 1st frame in the video
        if i == skip_frames:
            output_1st_img, pose_1st_img = detector(
                img,
                args.detect_resolution,
                args.image_resolution,
                output_type="cv2",
                return_pose_dict=True,
            )
            body_1st_img = pose_1st_img["bodies"]["candidate"]
            subset_1st_img = pose_1st_img["bodies"]["subset"][0]
            hands_1st_img = pose_1st_img["hands"]
            faces_1st_img = pose_1st_img["faces"]

            """
            计算逻辑:
            1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
            2. 用点在图中的实际坐标来计算。
            3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
            4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
            注意：dwpose 输出是 (w, h)
            """

            # h不变，w缩放到原比例
            ref_H, ref_W = refer_img.shape[0], refer_img.shape[1]
            ref_ratio = ref_W / ref_H
            body_ref_img[:, 0] = body_ref_img[:, 0] * ref_ratio
            hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
            faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

            video_ratio = width / height
            body_1st_img[:, 0] = body_1st_img[:, 0] * video_ratio
            hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
            faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

            # face
            align_args = dict()
            face_ref = faces_ref_img[0]
            face_1st = faces_1st_img[0]
            x1_ref, x2_ref = np.min(face_ref[:, 0]), np.max(face_ref[:, 0])
            y1_ref, y2_ref = np.min(face_ref[:, 1]), np.max(face_ref[:, 1])
            w_ref = x2_ref - x1_ref
            h_ref = y2_ref - y1_ref
            x1_1st, x2_1st = np.min(face_1st[:, 0]), np.max(face_1st[:, 0])
            y1_1st, y2_1st = np.min(face_1st[:, 1]), np.max(face_1st[:, 1])
            w_1st = x2_1st - x1_1st
            h_1st = y2_1st - y1_1st
            align_args["scale_face_w"] = w_ref / w_1st
            align_args["scale_face_h"] = h_ref / h_1st
            align_args["scale_neck"] = h_ref / h_1st
            align_args["scale_face"] = (h_ref / h_1st + w_ref / w_1st) / 2
            align_args["scale_shoulder"] = w_ref / w_1st
            align_args["scale_arm_upper"] = (h_ref / h_1st + w_ref / w_1st) / 2
            align_args["scale_arm_lower"] = (h_ref / h_1st + w_ref / w_1st) / 2
            align_args["scale_hand"] = (h_ref / h_1st + w_ref / w_1st) / 2
            align_args["scale_leg_upper"] = (h_ref / h_1st + w_ref / w_1st) / 2
            align_args["scale_leg_lower"] = (h_ref / h_1st + w_ref / w_1st) / 2
            align_args["scale_body_len"] = h_ref / h_1st

            if -1 not in (
                subset_1st_img[0],
                subset_1st_img[1],
                subset_ref_img[0],
                subset_ref_img[1],
            ):
                dist_1st_img = np.linalg.norm(
                    body_1st_img[0] - body_1st_img[1]
                )  # 0.078
                dist_ref_img = np.linalg.norm(
                    body_ref_img[0] - body_ref_img[1]
                )  # 0.106
                align_args["scale_neck"] = (
                    dist_ref_img / dist_1st_img
                )  # align / pose = ref / 1st

            # if -1 not in (subset_1st_img[16], subset_1st_img[17], subset_ref_img[16], subset_ref_img[17]):
            #     dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
            #     dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
            #     align_args["scale_face"] = dist_ref_img / dist_1st_img

            if -1 not in (
                subset_1st_img[2],
                subset_1st_img[5],
                subset_ref_img[2],
                subset_ref_img[5],
            ):
                dist_1st_img = np.linalg.norm(
                    body_1st_img[2] - body_1st_img[5]
                )  # 0.112
                dist_ref_img = np.linalg.norm(
                    body_ref_img[2] - body_ref_img[5]
                )  # 0.174
                align_args["scale_shoulder"] = dist_ref_img / dist_1st_img

            if -1 not in (
                subset_1st_img[2],
                subset_1st_img[3],
                subset_1st_img[5],
                subset_1st_img[6],
                subset_ref_img[2],
                subset_ref_img[3],
                subset_ref_img[5],
                subset_ref_img[6],
            ):
                dist_1st_img1 = np.linalg.norm(
                    body_1st_img[2] - body_1st_img[3]
                )  # 0.895
                dist_ref_img1 = np.linalg.norm(
                    body_ref_img[2] - body_ref_img[3]
                )  # 0.134
                dist_1st_img2 = np.linalg.norm(body_1st_img[5] - body_1st_img[6])
                dist_ref_img2 = np.linalg.norm(body_ref_img[5] - body_ref_img[6])
                align_args["scale_arm_upper"] = max(dist_ref_img1, dist_ref_img2) / max(
                    dist_1st_img1, dist_1st_img2
                )
                # align_args["scale_arm_upper"] = (dist_ref_img1 / dist_1st_img1 + dist_ref_img2 / dist_1st_img2) / 2

            if -1 not in (
                subset_1st_img[3],
                subset_1st_img[4],
                subset_1st_img[6],
                subset_1st_img[7],
                subset_ref_img[2],
                subset_ref_img[3],
                subset_ref_img[5],
                subset_ref_img[6],
            ):
                dist_1st_img1 = np.linalg.norm(body_1st_img[3] - body_1st_img[4])
                dist_ref_img1 = np.linalg.norm(body_ref_img[3] - body_ref_img[4])
                dist_1st_img2 = np.linalg.norm(body_1st_img[6] - body_1st_img[7])
                dist_ref_img2 = np.linalg.norm(body_ref_img[6] - body_ref_img[7])
                align_args["scale_arm_lower"] = max(dist_ref_img1, dist_ref_img2) / max(
                    dist_1st_img1, dist_1st_img2
                )
                # align_args["scale_arm_lower"] = (dist_ref_img1 / dist_1st_img1 + dist_ref_img2 / dist_1st_img2) / 2

            # hand
            dist_1st_img = np.zeros(10)
            dist_ref_img = np.zeros(10)

            dist_1st_img[0] = np.linalg.norm(hands_1st_img[0, 0] - hands_1st_img[0, 1])
            dist_1st_img[1] = np.linalg.norm(hands_1st_img[0, 0] - hands_1st_img[0, 5])
            dist_1st_img[2] = np.linalg.norm(hands_1st_img[0, 0] - hands_1st_img[0, 9])
            dist_1st_img[3] = np.linalg.norm(hands_1st_img[0, 0] - hands_1st_img[0, 13])
            dist_1st_img[4] = np.linalg.norm(hands_1st_img[0, 0] - hands_1st_img[0, 17])
            dist_1st_img[5] = np.linalg.norm(hands_1st_img[1, 0] - hands_1st_img[1, 1])
            dist_1st_img[6] = np.linalg.norm(hands_1st_img[1, 0] - hands_1st_img[1, 5])
            dist_1st_img[7] = np.linalg.norm(hands_1st_img[1, 0] - hands_1st_img[1, 9])
            dist_1st_img[8] = np.linalg.norm(hands_1st_img[1, 0] - hands_1st_img[1, 13])
            dist_1st_img[9] = np.linalg.norm(hands_1st_img[1, 0] - hands_1st_img[1, 17])

            dist_ref_img[0] = np.linalg.norm(hands_ref_img[0, 0] - hands_ref_img[0, 1])
            dist_ref_img[1] = np.linalg.norm(hands_ref_img[0, 0] - hands_ref_img[0, 5])
            dist_ref_img[2] = np.linalg.norm(hands_ref_img[0, 0] - hands_ref_img[0, 9])
            dist_ref_img[3] = np.linalg.norm(hands_ref_img[0, 0] - hands_ref_img[0, 13])
            dist_ref_img[4] = np.linalg.norm(hands_ref_img[0, 0] - hands_ref_img[0, 17])
            dist_ref_img[5] = np.linalg.norm(hands_ref_img[1, 0] - hands_ref_img[1, 1])
            dist_ref_img[6] = np.linalg.norm(hands_ref_img[1, 0] - hands_ref_img[1, 5])
            dist_ref_img[7] = np.linalg.norm(hands_ref_img[1, 0] - hands_ref_img[1, 9])
            dist_ref_img[8] = np.linalg.norm(hands_ref_img[1, 0] - hands_ref_img[1, 13])
            dist_ref_img[9] = np.linalg.norm(hands_ref_img[1, 0] - hands_ref_img[1, 17])

            ratio = 0
            count = 0
            for i in range(10):
                if dist_1st_img[i] != 0:
                    ratio = ratio + dist_ref_img[i] / dist_1st_img[i]
                    count = count + 1
            if count != 0:
                align_args["scale_hand"] = (
                    ratio / count
                    + align_args["scale_arm_upper"]
                    + align_args["scale_arm_lower"]
                ) / 3
            else:
                align_args["scale_hand"] = (
                    align_args["scale_arm_upper"] + align_args["scale_arm_lower"]
                ) / 2

            # body
            if -1 not in (
                subset_1st_img[1],
                subset_1st_img[8],
                subset_1st_img[11],
                subset_ref_img[1],
                subset_ref_img[8],
                subset_ref_img[11],
            ):
                dist_1st_img = np.linalg.norm(
                    body_1st_img[1] - (body_1st_img[8] + body_1st_img[11]) / 2
                )
                dist_ref_img = np.linalg.norm(
                    body_ref_img[1] - (body_ref_img[8] + body_ref_img[11]) / 2
                )
                align_args["scale_body_len"] = dist_ref_img / dist_1st_img

            if -1 not in (
                subset_1st_img[8],
                subset_1st_img[9],
                subset_1st_img[11],
                subset_1st_img[12],
                subset_ref_img[8],
                subset_ref_img[9],
                subset_ref_img[11],
                subset_ref_img[12],
            ):
                dist_1st_img1 = np.linalg.norm(body_1st_img[8] - body_1st_img[9])
                dist_ref_img1 = np.linalg.norm(body_ref_img[8] - body_ref_img[9])
                dist_1st_img2 = np.linalg.norm(body_1st_img[11] - body_1st_img[12])
                dist_ref_img2 = np.linalg.norm(body_ref_img[11] - body_ref_img[12])
                align_args["scale_leg_upper"] = max(dist_ref_img1, dist_ref_img2) / max(
                    dist_1st_img1, dist_1st_img2
                )
                # align_args["scale_leg_upper"] = (dist_ref_img1 / dist_1st_img1 + dist_ref_img2 / dist_1st_img2) / 2

            if -1 not in (
                subset_1st_img[9],
                subset_1st_img[10],
                subset_1st_img[12],
                subset_1st_img[13],
                subset_ref_img[9],
                subset_ref_img[10],
                subset_ref_img[12],
                subset_ref_img[13],
            ):
                dist_1st_img1 = np.linalg.norm(body_1st_img[9] - body_1st_img[10])
                dist_ref_img1 = np.linalg.norm(body_ref_img[9] - body_ref_img[10])
                dist_1st_img2 = np.linalg.norm(body_1st_img[12] - body_1st_img[13])
                dist_ref_img2 = np.linalg.norm(body_ref_img[12] - body_ref_img[13])
                align_args["scale_leg_lower"] = max(dist_ref_img1, dist_ref_img2) / max(
                    dist_1st_img1, dist_1st_img2
                )
                # align_args["scale_leg_lower"] = (dist_ref_img1 / dist_1st_img1 + dist_ref_img2 / dist_1st_img2) / 2

            ####################
            # need adjust nan
            for k, v in align_args.items():
                if np.isnan(v):
                    align_args[k] = 1

            # centre offset (the offset of key point 1)
            offset = body_ref_img[1] - body_1st_img[1]

        # pose align
        pose_img, pose_ori = detector(
            img,
            args.detect_resolution,
            args.image_resolution,
            output_type="cv2",
            return_pose_dict=True,
        )
        video_pose_buffer.append(pose_img)
        pose_align = align_img(img, pose_ori, align_args)

        # add centre offset
        pose = pose_align
        pose["bodies"]["candidate"] = pose["bodies"]["candidate"] + offset
        pose["hands"] = pose["hands"] + offset
        pose["faces"] = pose["faces"] + offset

        # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
        pose["bodies"]["candidate"][:, 0] = (
            pose["bodies"]["candidate"][:, 0] / ref_ratio
        )
        pose["hands"][:, :, 0] = pose["hands"][:, :, 0] / ref_ratio
        pose["faces"][:, :, 0] = pose["faces"][:, :, 0] / ref_ratio
        pose_list.append(pose)

    # stack
    body_list = [pose["bodies"]["candidate"][:18] for pose in pose_list]
    body_list_subset = [pose["bodies"]["subset"][:1] for pose in pose_list]
    hands_list = [pose["hands"][:2] for pose in pose_list]
    faces_list = [pose["faces"][:1] for pose in pose_list]

    body_seq = np.stack(body_list, axis=0)
    body_seq_subset = np.stack(body_list_subset, axis=0)
    hands_seq = np.stack(hands_list, axis=0)
    faces_seq = np.stack(faces_list, axis=0)

    # concatenate and paint results
    H = 768  # paint height
    W1 = int((H / ref_H * ref_W) // 2 * 2)
    W2 = int((H / height * width) // 2 * 2)
    result_demo = []  # = Writer(args, None, H, 3*W1+2*W2, demo_out, fps)
    result_pose_only = []  # Writer(args, None, H, W1, pose_out, fps)
    for i in range(len(body_seq)):
        pose_t = {}
        pose_t["bodies"] = {}
        pose_t["bodies"]["candidate"] = body_seq[i]
        pose_t["bodies"]["subset"] = body_seq_subset[i]
        pose_t["hands"] = hands_seq[i]
        pose_t["faces"] = faces_seq[i]

        ref_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)
        ref_img = cv2.resize(ref_img, (W1, H))
        ref_pose = cv2.resize(output_refer, (W1, H))

        output_transformed = draw_pose(
            pose_t,
            int(H_in * 1024 / W_in),
            1024,
            draw_face=args.face,
            draw_hand=args.hand,
        )
        output_transformed = cv2.cvtColor(output_transformed, cv2.COLOR_BGR2RGB)
        output_transformed = cv2.resize(output_transformed, (W1, H))

        video_frame = cv2.resize(video_frame_buffer[i], (W2, H))
        video_pose = cv2.resize(video_pose_buffer[i], (W2, H))

        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_pose = cv2.cvtColor(video_pose, cv2.COLOR_BGR2RGB)

        res = np.concatenate(
            [ref_img, ref_pose, output_transformed, video_frame, video_pose], axis=1
        )
        result_demo.append(res)
        result_pose_only.append(output_transformed)

    print(f"pose_list len: {len(pose_list)}")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(result_demo, fps=fps)
    clip.write_videofile(demo_out, fps=fps)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
        result_pose_only, fps=fps
    )
    clip.write_videofile(pose_out, fps=fps)
    print("pose align done")


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.outfn is None:
        time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_dir = os.path.dirname(os.path.abspath(__file__))
        args.outfn = os.path.join(file_dir, f"assets/{time}")

    os.makedirs(args.outfn, exist_ok=True)

    img_name = os.path.basename(args.imgfn)
    img_name = os.path.splitext(img_name)[0]
    video_name = os.path.basename(args.vidfn)
    video_name = os.path.splitext(video_name)[0]

    demo_out = os.path.join(args.outfn, f"img_{img_name}_video_{video_name}_demo.mp4")
    pose_out = os.path.join(args.outfn, f"img_{img_name}_video_{video_name}_pose.mp4")

    run_align_video_with_filterPose_translate_smooth(
        args, demo_out=demo_out, pose_out=pose_out
    )


if __name__ == "__main__":
    main()
