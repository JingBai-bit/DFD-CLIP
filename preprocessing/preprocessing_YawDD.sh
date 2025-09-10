#!/bin/bash

# 配置参数
VIDEO_ROOT="/home/baijing/data/datasets/YawDD/Mirror"
MALE_VIDEO_DIR="$VIDEO_ROOT/Male_mirror"
FEMALE_VIDEO_DIR="$VIDEO_ROOT/Female_mirror"
OUTPUT_ROOT="/home/baijing/data/DFD-CLIP/datasets/YawDD_Frame_Face"
TRAIN_RATIO=0.8  # 训练集比例
RANDOM_SEED=42   # 随机种子保证可重复性

# 创建目录结构
mkdir -p "$OUTPUT_ROOT"/{train,test}
TRAIN_FILE="$OUTPUT_ROOT/train_annotations.txt"
TEST_FILE="$OUTPUT_ROOT/test_annotations.txt"
> "$TRAIN_FILE"
> "$TEST_FILE"

# 标签映射（移除了Talking&Yawning相关标签）
declare -A LABEL_MAP=(
    ["Normal"]=0
    ["Yawning"]=1
)

# 收集所有视频文件（跳过不需要的标签）
ALL_VIDEOS=()
for video in "$MALE_VIDEO_DIR"/*.avi "$FEMALE_VIDEO_DIR"/*.avi; do
    video_name=$(basename "$video" .avi)
    label=$(echo "$video_name" | awk -F'-' '{print $NF}')

    # 跳过包含"Talking&Yawning"或"Talking&yawning"的视频
    if [[ "$label" == "Talking&Yawning" || "$label" == "Talking&yawning"|| "$label" == "Talking" ]]; then
        echo "Skipping video with unwanted label: $video"
        continue
    fi

    # 只处理在LABEL_MAP中定义的标签
    if [[ -n "${LABEL_MAP[$label]}" ]]; then
        ALL_VIDEOS+=("$video")
    else
        echo "Skipping video with unknown label: $video"
    fi
done

# 随机打乱视频顺序
shuf_arrays() {
    local i tmp size max rand
    size=${#ALL_VIDEOS[@]}
    max=$(( 32768 / size * size ))
    for ((i=size-1; i>0; i--)); do
        while (( (rand=RANDOM) >= max )); do :; done
        rand=$(( rand % (i+1) ))
        tmp=${ALL_VIDEOS[i]}
        ALL_VIDEOS[i]=${ALL_VIDEOS[rand]}
        ALL_VIDEOS[rand]=$tmp
    done
}
shuf_arrays

# 计算分割点
total=${#ALL_VIDEOS[@]}
train_num=$(awk "BEGIN {print int($total*$TRAIN_RATIO)}")

# 处理函数
process_video() {
    local video_path=$1
    local output_dir=$2
    local video_name=$(basename "$video_path" .avi)
    local label=$(echo "$video_name" | awk -F'-' '{print $NF}')

    # 提取帧
    mkdir -p "$output_dir"
    ffmpeg -i "$video_path" -r 5 -vf "scale=224:224" -q:v 2 "$output_dir/%05d.jpg" 2>/dev/null

    # 返回标注信息
    local frame_count=$(ls -1 "$output_dir" | wc -l)
    echo "$output_dir $frame_count ${LABEL_MAP[$label]}"
}

# 主处理循环
counter=1
for ((i=0; i<total; i++)); do
    video_path="${ALL_VIDEOS[i]}"
    video_name=$(basename "$video_path" .avi)
    label=$(echo "$video_name" | awk -F'-' '{print $NF}')

    # 再次检查标签（冗余保护）
    if [[ "$label" == "Talking&Yawning" || "$label" == "Talking&yawning" ]]; then
        echo "Warning: Unexpectedly processing skipped label: $video_path"
        continue
    fi

    if (( i < train_num )); then
        # 训练集
        output_dir="$OUTPUT_ROOT/train/$(printf "%05d" $counter)"
        annotation=$(process_video "$video_path" "$output_dir")
        echo "$annotation" >> "$TRAIN_FILE"
    else
        # 测试集
        output_dir="$OUTPUT_ROOT/test/$(printf "%05d" $counter)"
        annotation=$(process_video "$video_path" "$output_dir")
        echo "$annotation" >> "$TEST_FILE"
    fi

    echo "Processed [$counter/$total]: $video_path -> ${annotation%% *}"
    ((counter++))
done

# 生成统计信息
echo "=== Dataset Split Summary ==="
echo "Total valid videos: $total"
echo "Training set: $(wc -l < "$TRAIN_FILE") (${TRAIN_RATIO*100}%)"
echo "Test set:     $(wc -l < "$TEST_FILE") ($((100-TRAIN_RATIO*100))%)"
echo "Output:"
echo " - Train annotations: $TRAIN_FILE"
echo " - Test annotations:  $TEST_FILE"
echo " - Frames saved to:   $OUTPUT_ROOT/{train,test}"