#TODO: Please refer to run_problem1.sh for the content of this file
# Usage:
# bash scripts/run_problem1.sh --config cnn_unet

CONFIG=$2

PUBLIC_DIR="dataset/public"
PRIVATE_DIR="dataset/private"

echo "Generating submission using config: $CONFIG"
echo "Public dir: $PUBLIC_DIR"
echo "Private dir: $PRIVATE_DIR"

uv run generate_submission.py \
    --config $CONFIG \
    --public_dir $PUBLIC_DIR \
    --private_dir $PRIVATE_DIR

echo "✅ Submission saved to submission.csv"
