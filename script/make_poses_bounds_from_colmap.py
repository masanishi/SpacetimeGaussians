#!/usr/bin/env python3
"""
LLFFのposes_bounds.npyをCOLMAPのundistorted出力から生成する簡易スクリプト。

前提:
<scene>/
  ├─ images/           # undistorted images
  └─ sparse/0/
       cameras.txt
       images.txt
       points3D.txt

出力:
  <scene>/poses_bounds.npy

注意:
- ここでは深度境界(bounds)はCOLMAP点群の最近/最遠距離の分位点で近似する（5%と95%）。
- intrinsicsはPINHOLE fx=fyと仮定。異なる場合は近似する。

使い方:
  python script/make_poses_bounds_from_colmap.py --scene_dir <path_to_colmap_0>
"""
import os
import argparse
import numpy as np

def read_cameras_txt(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            el = line.strip().split()
            cam_id = int(el[0])
            model = el[1]
            w = float(el[2]); h = float(el[3])
            if model in ("PINHOLE", "SIMPLE_PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
                fx = float(el[4])
                fy = float(el[5]) if model == "PINHOLE" else fx
                cx = float(el[6]) if model == "PINHOLE" else float(el[5])
                cy = float(el[7]) if model == "PINHOLE" else float(el[6])
            else:
                # 未対応モデルは簡易にPINHOLE近似
                fx = float(el[4]); fy = fx
                cx = w/2; cy = h/2
            cams[cam_id] = dict(model=model, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams

def read_images_txt(path):
    imgs = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            el = line.strip().split()
            if len(el) < 10:
                continue
            image_id = int(el[0])
            qw, qx, qy, qz = map(float, el[1:5])
            tx, ty, tz = map(float, el[5:8])
            cam_id = int(el[8])
            name = el[9]
            imgs.append(dict(id=image_id, q=np.array([qw,qx,qy,qz]), t=np.array([tx,ty,tz]), cam_id=cam_id, name=name))
    # images.txtは2行構成(次行が2D点)だが、ここでは1行目だけ読む
    return imgs

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def compose_llff_pose(R, t):
    # COLMAP: world->cam: [R | t]; 変換: cam2world = [R.T | -R.T t]
    c2w_R = R.T
    c2w_t = -c2w_R @ t
    # LLFF pose (3x5): [R|t|[h w f]]^T 形式に合わせて後で補う
    return c2w_R, c2w_t

def estimate_bounds(c2w_list):
    # 単純化: 平均カメラ中心からの距離分布で近/遠を推定
    centers = np.stack([t for _, t in c2w_list], axis=0)
    dists = np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=1)
    near = np.quantile(dists, 0.05)
    far = np.quantile(dists, 0.95)
    return near, far

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', type=str, required=True, help='colmap_x/ ディレクトリ (undistorted出力を含む)')
    args = ap.parse_args()

    sparse = os.path.join(args.scene_dir, 'sparse', '0')
    cam_p = os.path.join(sparse, 'cameras.txt')
    img_p = os.path.join(sparse, 'images.txt')
    # .txt が無い場合は .bin から変換を試みる
    if not (os.path.exists(cam_p) and os.path.exists(img_p)):
        cam_bin = os.path.join(sparse, 'cameras.bin')
        img_bin = os.path.join(sparse, 'images.bin')
        if os.path.exists(cam_bin) and os.path.exists(img_bin):
            print('TXTが無いため、COLMAP model_converterでTXTへ変換を試みます...')
            import subprocess, tempfile, shutil
            with tempfile.TemporaryDirectory() as td:
                try:
                    subprocess.check_call([
                        'colmap', 'model_converter',
                        '--input_path', sparse,
                        '--output_path', td,
                        '--output_type', 'TXT'
                    ])
                    # 生成物をsparse/0にコピー
                    for fn in ('cameras.txt','images.txt','points3D.txt'):
                        s = os.path.join(td, fn)
                        if os.path.exists(s):
                            shutil.copy2(s, os.path.join(sparse, fn))
                except Exception as e:
                    raise FileNotFoundError('cameras.txt / images.txt に変換できませんでした。colmap がPATHにあるか確認してください。') from e
        if not (os.path.exists(cam_p) and os.path.exists(img_p)):
            raise FileNotFoundError('cameras.txt / images.txt が見つかりません。image_undistorter 後のフォルダを指定してください。')

    cams = read_cameras_txt(cam_p)
    imgs = read_images_txt(img_p)

    # 名前順で安定化
    imgs = sorted(imgs, key=lambda x: x['name'])

    poses_arr = []
    c2w_list = []
    for im in imgs:
        R = qvec2rotmat(im['q'])
        t = im['t']
        c2w_R, c2w_t = compose_llff_pose(R, t)
        cam = cams[im['cam_id']]
        H, W = cam['h'], cam['w']
        f = (cam['fx'] + cam['fy']) / 2.0
        pose = np.concatenate([c2w_R, c2w_t.reshape(3,1), np.array([[H],[W],[f]])], axis=1)  # 3x5
        poses_arr.append(pose)
        c2w_list.append((c2w_R, c2w_t))

    poses = np.stack(poses_arr, axis=2)  # 3x5xN
    near, far = estimate_bounds(c2w_list)
    bounds = np.tile(np.array([[near],[far]]), (1, poses.shape[2]))  # 2xN

    # LLFF形式: N x 17 (3x5=15 + 2 bounds)
    poses_bounds = np.concatenate([poses.transpose(2,0,1).reshape(-1,15), bounds.T], axis=1)
    out = os.path.join(os.path.dirname(args.scene_dir), 'poses_bounds.npy') if os.path.basename(args.scene_dir).startswith('colmap_') else os.path.join(args.scene_dir, 'poses_bounds.npy')
    np.save(out, poses_bounds)
    print(f"Saved poses_bounds.npy -> {out} (N={poses.shape[2]})")

if __name__ == '__main__':
    main()
