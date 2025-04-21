#!/bin/bash

# YOLOv5检测脚本 - 集中管理模型权重
# 作者: Cursor AI
# 版本: 1.0

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 项目根目录
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
# 权重目录
WEIGHTS_DIR="$ROOT_DIR/weights"

# 默认设置
DEFAULT_MODEL="yolov5s"
DEFAULT_CONF=0.25
AVAILABLE_MODELS=("yolov5n" "yolov5s" "yolov5m" "yolov5l" "yolov5x" "yolov5n6" "yolov5s6" "yolov5m6" "yolov5l6" "yolov5x6")

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 显示帮助信息
show_help() {
    echo -e "${BLUE}YOLOv5检测脚本${NC} - 集中管理模型权重"
    echo ""
    echo "用法: $(basename $0) [选项] --source <输入源>"
    echo ""
    echo "选项:"
    echo "  -h, --help                显示此帮助信息"
    echo "  -m, --model <模型名称>    指定YOLOv5模型 (默认: $DEFAULT_MODEL)"
    echo "  -l, --list                列出可用的模型"
    echo "  --weights-dir <目录路径>  指定自定义权重目录 (默认: $WEIGHTS_DIR)"
    echo ""
    echo "可用模型:"
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo "  - $model"
    done
    echo ""
    echo "示例:"
    echo "  $(basename $0) --source data/images/bus.jpg"
    echo "  $(basename $0) --model yolov5m --source data/images/zidane.jpg --conf 0.4"
    echo ""
    echo "注意: 所有未识别的参数将直接传递给YOLOv5的detect.py脚本"
    echo ""
}

# 列出可用模型
list_models() {
    echo -e "${BLUE}可用的YOLOv5模型:${NC}"
    for model in "${AVAILABLE_MODELS[@]}"; do
        if [ -f "$WEIGHTS_DIR/${model}.pt" ]; then
            echo -e "  - ${GREEN}$model${NC} (已下载)"
        else
            echo -e "  - ${YELLOW}$model${NC} (未下载)"
        fi
    done
}

# 检查模型是否有效
is_valid_model() {
    local model_name=$1
    for model in "${AVAILABLE_MODELS[@]}"; do
        if [ "$model" = "$model_name" ]; then
            return 0
        fi
    done
    return 1
}

# 下载模型权重
download_model() {
    local model_name=$1
    local model_file="${model_name}.pt"
    local model_path="$WEIGHTS_DIR/$model_file"
    
    if [ ! -f "$model_path" ]; then
        echo -e "${YELLOW}正在下载 ${model_name} 模型权重...${NC}"
        mkdir -p "$WEIGHTS_DIR"
        
        # 从GitHub下载模型
        if ! wget -q --show-progress https://github.com/ultralytics/yolov5/releases/download/v7.0/${model_file} -O "$model_path"; then
            echo -e "${RED}下载失败: ${model_name}${NC}"
            rm -f "$model_path" # 删除可能的部分下载文件
            return 1
        fi
        
        echo -e "${GREEN}模型下载完成: ${model_name}${NC}"
    else
        echo -e "${GREEN}使用本地模型: ${model_name}${NC}"
    fi
    
    return 0
}

# 主函数
main() {
    # 解析参数
    MODEL="$DEFAULT_MODEL"
    CUSTOM_WEIGHTS_DIR=""
    DETECT_ARGS=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--model)
                MODEL="$2"
                shift 2
                ;;
            -l|--list)
                list_models
                exit 0
                ;;
            --weights-dir)
                CUSTOM_WEIGHTS_DIR="$2"
                shift 2
                ;;
            *)
                DETECT_ARGS+=("$1")
                shift
                ;;
        esac
    done
    
    # 如果指定了自定义权重目录，使用它
    if [ -n "$CUSTOM_WEIGHTS_DIR" ]; then
        WEIGHTS_DIR="$CUSTOM_WEIGHTS_DIR"
    fi
    
    # 检查模型名称是否有效
    if ! is_valid_model "$MODEL"; then
        echo -e "${RED}错误: 无效的模型 '$MODEL'${NC}"
        echo "请使用以下模型之一:"
        for model in "${AVAILABLE_MODELS[@]}"; do
            echo "  - $model"
        done
        exit 1
    fi
    
    # 下载模型权重（如果需要）
    if ! download_model "$MODEL"; then
        echo -e "${RED}无法获取模型权重，退出。${NC}"
        exit 1
    fi
    
    # 构建权重路径
    WEIGHTS_PATH="$WEIGHTS_DIR/${MODEL}.pt"
    
    # 运行检测
    echo -e "${BLUE}运行YOLOv5检测...${NC}"
    echo -e "模型: ${GREEN}${MODEL}${NC}"
    echo -e "权重路径: ${WEIGHTS_PATH}"
    echo -e "参数: ${DETECT_ARGS[@]}"
    echo ""
    
    cd "$ROOT_DIR"
    python detect.py --weights "$WEIGHTS_PATH" "${DETECT_ARGS[@]}"
    
    echo -e "${GREEN}检测完成!${NC}"
}

# 执行主函数
main "$@" 