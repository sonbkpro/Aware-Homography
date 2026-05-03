Evaluation data directory — matches Zhang et al. (ECCV 2020) protocol.

Expected structure (5 scene categories):
    data/eval/
        RE/              Regular (plain) scenes
            frame_a/     0001.png  0002.png  ...
            frame_b/     0001.png  0002.png  ...
            gt_points/   0001.npy  ...   (optional, shape: N×2×2 [src_xy, dst_xy])
        LT/              Low-texture scenes
        LL/              Low-light scenes
        SF/              Small foreground
        LF/              Large foreground

To prepare from your own videos:
    python scripts/prepare_eval_data.py \
        --video my_video.mp4 \
        --out_dir data/eval/MY_CATEGORY \
        --stride 5 --num_pairs 200
