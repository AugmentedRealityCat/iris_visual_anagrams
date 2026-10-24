name="patch.lemur.kangaroo"
python generate.py --name ${name} --prompts "a lemur" "a kangaroo" --style "a pencil sketch of" --views identity patch_permute --num_samples 1 --num_inference_steps 4 --guidance_scale 1.0 --generate_1024 --save_dir results/test
python animate.py --im_path results/test/${name}/0000/sample_1024.png --metadata_path results/test/${name}/metadata.pkl