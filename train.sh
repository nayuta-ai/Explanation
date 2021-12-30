TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
TMP_RESULTS_DIR="$(pwd)/.tmp_results/${TIMESTAMP}"
ARGS_FILE="${TMP_RESULTS_DIR}/args.yaml"
TRAIN_LOG_FILE="${TMP_RESULTS_DIR}/log.txt"
mkdir -p $TMP_RESULTS_DIR

export CUDA_VISIBLE_DEVICES=0,1
nohup python -u train.py \
    --args_file $ARGS_FILE \
    --tmp_results_dir $TMP_RESULTS_DIR \
    --train_log_file_path $TRAIN_LOG_FILE \
    > $TRAIN_LOG_FILE &

tail -f $TRAIN_LOG_FILE