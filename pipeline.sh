#!/bin/bash

ROOT_DIR="$PWD"
DATA_DIR="$ROOT_DIR"/data

SRC_FACE_DIR="$DATA_DIR"/src_face
SRC_BODY_DIR="$DATA_DIR"/src_body

SRC_BG_REMOVAL_DIR="$DATA_DIR"/bg_src
DST_BG_REMOVAL_DIR="$DATA_DIR"/bg_dst
rm -rf "$SRC_BG_REMOVAL_DIR"
rm -rf "$DST_BG_REMOVAL_DIR"
mkdir -p "$SRC_BG_REMOVAL_DIR"
mkdir -p "$DST_BG_REMOVAL_DIR"

SRC_HEAD_DETECTION_DIR="$DATA_DIR"/head_det_src
DST_HEAD_DETECTION_DIR="$DATA_DIR"/head_det_dst
rm -rf "$SRC_HEAD_DETECTION_DIR"
rm -rf "$DST_HEAD_DETECTION_DIR"
mkdir -p "$SRC_HEAD_DETECTION_DIR"
mkdir -p "$DST_HEAD_DETECTION_DIR"

SRC_INPAINTING_MASK_DIR="$DATA_DIR"/inpainting_src/mask
SRC_INPAINTING_FACE_DIR="$DATA_DIR"/inpainting_src/face
DST_INPAINTING_DIR="$DATA_DIR"/inpainting_dst
rm -rf "$SRC_INPAINTING_MASK_DIR"
rm -rf "$SRC_INPAINTING_FACE_DIR"
rm -rf "$DST_INPAINTING_DIR"
mkdir -p "$SRC_INPAINTING_MASK_DIR"
mkdir -p "$SRC_INPAINTING_FACE_DIR"
mkdir -p "$DST_INPAINTING_DIR"

SRC_HUMAN_PARSING_DIR="$DATA_DIR"/human_parsing_src
DST_HUMAN_PARSING_DIR_ATR="$DATA_DIR"/human_parsing_dst_atr
DST_HUMAN_PARSING_DIR_LIP="$DATA_DIR"/human_parsing_dst_lip
rm -rf "$SRC_HUMAN_PARSING_DIR"
rm -rf "$DST_HUMAN_PARSING_DIR_ATR"
rm -rf "$DST_HUMAN_PARSING_DIR_LIP"
mkdir -p "$SRC_HUMAN_PARSING_DIR"
mkdir -p "$DST_HUMAN_PARSING_DIR_ATR"
mkdir -p "$DST_HUMAN_PARSING_DIR_LIP"

SRC_HUMAN_PARSING_HEAD_DIR="$DATA_DIR"/human_parsing_head_src
DST_HUMAN_PARSING_HEAD_DIR_ATR="$DATA_DIR"/human_parsing_head_dst_atr
DST_HUMAN_PARSING_HEAD_DIR_LIP="$DATA_DIR"/human_parsing_head_dst_lip
rm -rf "$SRC_HUMAN_PARSING_HEAD_DIR"
rm -rf "$DST_HUMAN_PARSING_HEAD_DIR_ATR"
rm -rf "$DST_HUMAN_PARSING_HEAD_DIR_LIP"
mkdir -p "$SRC_HUMAN_PARSING_HEAD_DIR"
mkdir -p "$DST_HUMAN_PARSING_HEAD_DIR_ATR"
mkdir -p "$DST_HUMAN_PARSING_HEAD_DIR_LIP"

SRC_SKIN_CORRECTION_FACE_IMAGE_DIR="$DATA_DIR"/skin_src/face/image
SRC_SKIN_CORRECTION_FACE_MASK_DIR="$DATA_DIR"/skin_src/face/mask
SRC_SKIN_CORRECTION_BODY_IMAGE_DIR="$DATA_DIR"/skin_src/body/image
SRC_SKIN_CORRECTION_BODY_MASK_DIR="$DATA_DIR"/skin_src/body/mask
DST_SKIN_CORRECTION_DIR="$DATA_DIR"/skin_dst
rm -rf "$SRC_SKIN_CORRECTION_FACE_IMAGE_DIR"
rm -rf "$SRC_SKIN_CORRECTION_FACE_MASK_DIR"
rm -rf "$SRC_SKIN_CORRECTION_BODY_IMAGE_DIR"
rm -rf "$SRC_SKIN_CORRECTION_BODY_MASK_DIR"
rm -rf "$DST_SKIN_CORRECTION_DIR"
mkdir -p "$SRC_SKIN_CORRECTION_FACE_IMAGE_DIR"
mkdir -p "$SRC_SKIN_CORRECTION_FACE_MASK_DIR"
mkdir -p "$SRC_SKIN_CORRECTION_BODY_IMAGE_DIR"
mkdir -p "$SRC_SKIN_CORRECTION_BODY_MASK_DIR"
mkdir -p "$DST_SKIN_CORRECTION_DIR"

SRC_FACE_POSE_FACE_DIR="$DATA_DIR"/face_pose_src/face
SRC_FACE_POSE_BODY_DIR="$DATA_DIR"/face_pose_src/body
DST_FACE_POSE_DIR="$DATA_DIR"/face_pose_dst
#rm -rf "$SRC_FACE_POSE_FACE_DIR"
#rm -rf "$SRC_FACE_POSE_BODY_DIR"
#rm -rf "$DST_FACE_POSE_DIR"
mkdir -p "$SRC_FACE_POSE_FACE_DIR"
mkdir -p "$SRC_FACE_POSE_BODY_DIR"
mkdir -p "$DST_FACE_POSE_DIR"

cd "$ROOT_DIR"

echo "background removal"
cp "$SRC_FACE_DIR"/* "$SRC_BG_REMOVAL_DIR"
pwd
cd src/background/backgroud_rem
ls
python back_rem.py -i "$SRC_BG_REMOVAL_DIR" -o "$DST_BG_REMOVAL_DIR" -q 512
cd -

echo "head detection"
cp "$SRC_BODY_DIR"/* "$SRC_HEAD_DETECTION_DIR"
cd src/background/get_head
python get_head_box.py -i "$SRC_HEAD_DETECTION_DIR" -o "$DST_HEAD_DETECTION_DIR"
cd -

echo "human parsing"
cp "$SRC_BODY_DIR"/* "$SRC_HUMAN_PARSING_DIR"
cd 3rdparty/Self-Correction-Human-Parsing/
mkdir models
if [ ! -f models/atr.pth ]; then
  gdown -O models/atr.pth https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP
fi
if [ ! -f models/lip.pth ]; then
  gdown -O models/lip.pth https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH
fi

python simple_extractor.py --dataset atr --model-restore models/atr.pth --input-dir "$SRC_HUMAN_PARSING_DIR" --output-dir "$DST_HUMAN_PARSING_DIR_ATR"
python simple_extractor.py --dataset lip --model-restore models/lip.pth --input-dir "$SRC_HUMAN_PARSING_DIR" --output-dir "$DST_HUMAN_PARSING_DIR_LIP"
cd -

#echo "head pose change"
#cp "$DST_BG_REMOVAL_DIR"/* "$SRC_FACE_POSE_FACE_DIR"
#cp "$DST_HEAD_DETECTION_DIR"/*.png "$SRC_FACE_POSE_BODY_DIR"
#cd src/head_pose
#HEAD_POSE_MODELS=models/pretrain
#mkdir -p "$HEAD_POSE_MODELS"
#if [ ! -f "$HEAD_POSE_MODELS"/styleganinv_ffhq256_encoder.pth ]; then
#  echo 'wget 1'
#  wget "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EXqix_JIEgtLl1FXI4uCkr8B5GPaiJyiLXL6cFbdcIKqEA?e=WYesel\&download\=1" -O "$HEAD_POSE_MODELS"/styleganinv_ffhq256_encoder.pth  --quiet
#fi
#if [ ! -f "$HEAD_POSE_MODELS"/styleganinv_ffhq256_generator.pth ]; then
#  echo 'wget 2'
#  wget "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EbuzMQ3ZLl1AqvKJzeeBq7IBoQD-C1LfMIC8USlmOMPt3Q?e=CMXn8W\&download\=1" -O "$HEAD_POSE_MODELS"/styleganinv_ffhq256_generator.pth  --quiet
#fi
#if [ ! -f "$HEAD_POSE_MODELS"/vgg16.pth ]; then
#  echo 'wget 3'
#  wget "https://mycuhk-my.sharepoint.com/:u:/g/personal/1155082926_link_cuhk_edu_hk/EQJUz9DInbxEnp0aomkGGzAB5b3ZZbtsOA-TXct9E4ONqA?e=smtO0T\&download\=1" -O "$HEAD_POSE_MODELS"/vgg16.pth  --quiet
#fi
#python pose_transfer.py --head_path "$SRC_FACE_POSE_FACE_DIR" --pose_path "$SRC_FACE_POSE_BODY_DIR" --dest_path "$DST_FACE_POSE_DIR"
#cd -

echo "human parsing head"
cp "$DST_BG_REMOVAL_DIR"/* "$SRC_HUMAN_PARSING_HEAD_DIR"
cd 3rdparty/Self-Correction-Human-Parsing/
mkdir models
if [ ! -f models/atr.pth ]; then
  gdown -O models/atr.pth https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP
fi
if [ ! -f models/lip.pth ]; then
  gdown -O models/lip.pth https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH
fi

python simple_extractor.py --dataset atr --model-restore models/atr.pth --input-dir "$SRC_HUMAN_PARSING_HEAD_DIR" --output-dir "$DST_HUMAN_PARSING_HEAD_DIR_ATR"
python simple_extractor.py --dataset lip --model-restore models/lip.pth --input-dir "$SRC_HUMAN_PARSING_HEAD_DIR" --output-dir "$DST_HUMAN_PARSING_HEAD_DIR_LIP"
cd -

echo "skin color correction"
cp "$DST_BG_REMOVAL_DIR"/* "$SRC_SKIN_CORRECTION_FACE_IMAGE_DIR"
cp "$DST_HUMAN_PARSING_HEAD_DIR_LIP"/* "$SRC_SKIN_CORRECTION_FACE_MASK_DIR"
cp "$SRC_BODY_DIR"/* "$SRC_SKIN_CORRECTION_BODY_IMAGE_DIR"
cp "$DST_HUMAN_PARSING_DIR_LIP"/* "$SRC_SKIN_CORRECTION_BODY_MASK_DIR"
#cp "$DST_HUMAN_PARSING_DIR_ATR"/* "$SRC_SKIN_CORRECTION_BODY_MASK_DIR"
echo
cd src/skin_color_correction/skin_color_correction
python skin_correct.py --head_img "$SRC_SKIN_CORRECTION_FACE_IMAGE_DIR" --head_mask "$SRC_SKIN_CORRECTION_FACE_MASK_DIR" --body_img "$SRC_SKIN_CORRECTION_BODY_IMAGE_DIR" --body_mask "$SRC_SKIN_CORRECTION_BODY_MASK_DIR" --output_dir "$DST_SKIN_CORRECTION_DIR"
cd -

echo "inpainting"
cp "$DST_SKIN_CORRECTION_DIR"/* "$SRC_INPAINTING_FACE_DIR"
cp "$DST_HUMAN_PARSING_DIR_LIP"/* "$SRC_INPAINTING_MASK_DIR"
cd src/background/inpaint
echo "$PYTHONPATH"
PYTHONPATH="$PWD"/src
python src/inpaint.py --dir_image "$SRC_INPAINTING_FACE_DIR" --dir_mask "$SRC_INPAINTING_MASK_DIR" --outputs "$DST_INPAINTING_DIR"
cd -

#echo "background removal"
#cp "$SRC_FACE_DIR"/* "$SRC_BG_REMOVAL_DIR"
#pwd
#cd src/background/backgroud_rem
#ls
#python back_rem.py -i "$SRC_BG_REMOVAL_DIR" -o "$DST_BG_REMOVAL_DIR" -q 512
#cd -

echo "neck removal"
rm -rf "$DATA_DIR"/neckless_src
rm -rf "$DATA_DIR"/neckless_dst
mkdir -p "$DATA_DIR"/neckless_src
mkdir -p "$DATA_DIR"/neckless_dst
cp "$DST_FACE_POSE_DIR"/* "$DATA_DIR"/neckless_src
cd 3rdparty/Self-Correction-Human-Parsing/
python simple_extractor.py --dataset lip --model-restore models/lip.pth --input-dir "$DATA_DIR"/neckless_src --output-dir "$DATA_DIR"/neckless_dst
cd -

echo "emplace face"
cd "$ROOT_DIR"
OUTPUT_DIR="output/3"
cp src/mask_labels.json mask_labels.json
python src/emplace_face.py --face "$DST_FACE_POSE_DIR" --body "$DST_SKIN_CORRECTION_DIR" --headless "$DST_INPAINTING_DIR" --dst "$OUTPUT_DIR" --face_mask "$DATA_DIR"/neckless_dst