#!/bin/bash

# 配置参数
DATA_ROOT="/home/baijing/data/datasets/NTHUDDD"
OUTPUT_ROOT="/home/baijing/data/DFD-CLIP/datasets/NTHUDDD_Frame_Face"
TRAIN_RATIO=0.8
RANDOM_SEED=42

# 标签映射
declare -A LABEL_MAP=(
    ["NotDrowsy"]=0
    ["Nodding"]=1
    ["SlowBlink"]=2
    ["Yawning"]=3
)

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"
TRAIN_FILE="$OUTPUT_ROOT/train_annotations.txt"
TEST_FILE="$OUTPUT_ROOT/test_annotations.txt"
> "$TRAIN_FILE"
> "$TEST_FILE"

# 按类别存储样本（使用分隔符）
declare -A CLASS_SAMPLES
for class_dir in "$DATA_ROOT"/*/; do
    class_name=$(basename "$class_dir")
    if [[ -n "${LABEL_MAP[$class_name]}" ]]; then
        CLASS_SAMPLES[$class_name]=""
        for sample_dir in "$class_dir"*/; do
            if [[ -d "$sample_dir" ]]; then
                img_count=$(find "$sample_dir" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
                if (( img_count > 0 )); then
                    CLASS_SAMPLES[$class_name]+="|$sample_dir $img_count ${LABEL_MAP[$class_name]}"
                fi
            fi
        done
        # 移除开头的分隔符
        CLASS_SAMPLES[$class_name]=${CLASS_SAMPLES[$class_name]#|}
    fi
done

# 对每个类别按比例划分
echo "Performing stratified sampling..."
for class_name in "${!CLASS_SAMPLES[@]}"; do
    # 分割字符串为数组
    IFS='|' read -ra samples <<< "${CLASS_SAMPLES[$class_name]}"
    num_samples=${#samples[@]}
    num_train=$(awk "BEGIN {print int($num_samples * $TRAIN_RATIO)}")

    # 打乱样本
    for ((i = num_samples - 1; i > 0; i--)); do
        j=$((RANDOM % (i + 1)))
        tmp=${samples[i]}
        samples[i]=${samples[j]}
        samples[j]=$tmp
    done

    # 写入文件
    for ((i=0; i<num_samples; i++)); do
        if ((i < num_train)); then
            echo "${samples[i]}" >> "$TRAIN_FILE"
        else
            echo "${samples[i]}" >> "$TEST_FILE"
        fi
    done
done

# 生成统计信息
echo "=== Dataset Summary (Stratified) ==="
echo "Class distributions:"
for class_name in "${!CLASS_SAMPLES[@]}"; do
    IFS='|' read -ra samples <<< "${CLASS_SAMPLES[$class_name]}"
    total=${#samples[@]}
    train_count=$(grep -c " ${LABEL_MAP[$class_name]}$" "$TRAIN_FILE")
    test_count=$(grep -c " ${LABEL_MAP[$class_name]}$" "$TEST_FILE")
    printf "%-10s: Total=%-4d Train=%-4d (%.1f%%) Test=%-4d (%.1f%%)\n" \
           "$class_name" "$total" "$train_count" \
           $(awk "BEGIN {print $train_count/$total*100}") \
           "$test_count" $(awk "BEGIN {print $test_count/$total*100}")
done