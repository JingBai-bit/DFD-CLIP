#!/bin/bash

# RLDD数据集预处理脚本

# 配置参数
VIDEO_ROOT="/home/baijing/data/datasets/RLDD"
OUTPUT_ROOT="/home/baijing/data/DFD-CLIP/datasets/RLDD_Frame_Face"
TRAIN_RATIO=0.8 # 训练集比例
RANDOM_SEED=42   # 随机种子
FRAME_RATE=5     # 帧提取率
RESOLUTION="224:224" # 分辨率

# 创建目录结构
mkdir -p "$OUTPUT_ROOT"/{train,val,test}
TRAIN_FILE="$OUTPUT_ROOT/train_annotations.txt"
VAL_FILE="$OUTPUT_ROOT/val_annotations.txt"
TEST_FILE="$OUTPUT_ROOT/test_annotations.txt"
> "$TRAIN_FILE"
> "$VAL_FILE"
> "$TEST_FILE"

# 标签映射
declare -A LABEL_MAP=(
    ["0"]=0  # Non-drowsy
    ["10"]=1 # Drowsy
)

# 收集所有视频文件
ALL_VIDEOS=()
for subject_dir in "$VIDEO_ROOT"/*/; do
    subject=$(basename "$subject_dir")
    for label_dir in "$subject_dir"/*/; do
        label=$(basename "$label_dir")

        # 只处理0和10标签的目录
        if [[ -n "${LABEL_MAP[$label]}" ]]; then
            for video in "$label_dir"/*.mp4; do
                if [[ -f "$video" ]]; then
                    ALL_VIDEOS+=("$video")
                fi
            done
        fi
    done
done

# 随机打乱视频顺序
echo "Total videos found: ${#ALL_VIDEOS[@]}"
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
val_num=$(awk "BEGIN {print int($total*$VAL_RATIO)}")

# 处理函数
process_video() {
    local video_path=$1
    local output_dir=$2
    local label_dir=$(dirname "$video_path")
    local label=$(basename "$label_dir")

    # 创建输出目录
    mkdir -p "$output_dir"

    # 提取帧 (添加-y覆盖已存在文件，-loglevel error减少输出)
    ffmpeg -i "$video_path" -r $FRAME_RATE -vf "scale=$RESOLUTION" -q:v 2 -loglevel error "$output_dir/%05d.jpg"

    # 返回标注信息
    local frame_count=$(ls -1 "$output_dir" | wc -l)
    echo "$output_dir $frame_count ${LABEL_MAP[$label]}"
}

# 主处理循环
counter=1
for ((i=0; i<total; i++)); do
    video_path="${ALL_VIDEOS[i]}"

    # 确定数据集分割
    if (( i < train_num )); then
        # 训练集
        output_dir="$OUTPUT_ROOT/train/$(printf "%05d" $counter)"
        output_file="$TRAIN_FILE"
    elif (( i < train_num + val_num )); then
        # 验证集
        output_dir="$OUTPUT_ROOT/val/$(printf "%05d" $counter)"
        output_file="$VAL_FILE"
    else
        # 测试集
        output_dir="$OUTPUT_ROOT/test/$(printf "%05d" $counter)"
        output_file="$TEST_FILE"
    fi

    # 处理视频
    annotation=$(process_video "$video_path" "$output_dir")
    echo "$annotation" >> "$output_file"

    # 显示进度
    printf "Processing [%d/%d] %s -> %s\n" "$counter" "$total" "$(basename "$video_path")" "${annotation%% *}"
    ((counter++))
done

# 生成统计信息
echo "=== Dataset Split Summary ==="
echo "Total videos processed: $total"
echo "Training set: $(wc -l < "$TRAIN_FILE") ($TRAIN_RATIO)"
echo "Validation set: $(wc -l < "$VAL_FILE") ($VAL_RATIO)"
echo "Test set: $(wc -l < "$TEST_FILE") ($((1-TRAIN_RATIO-VAL_RATIO)))"
echo "Output:"
echo " - Train annotations: $TRAIN_FILE"
echo " - Val annotations: $VAL_FILE"
echo " - Test annotations: $TEST_FILE"
echo " - Frames saved to: $OUTPUT_ROOT/{train,val,test}"