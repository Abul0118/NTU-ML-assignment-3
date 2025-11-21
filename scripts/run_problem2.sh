#TODO: Please refer to run_problem1.sh for the content of this file
#TODO: Please refer to run_problem1.sh for the content of this file
#!/bin/bash

# TODO: Download your checkpoint by gdown and unzip it here if needed

# Run the submission generation script
python generate_submission.py  --config transformer_unet --public_dir "${1}" --private_dir "${2}"
