#%%
from eval_metric import calculate_clip_image_similarity, calculate_fid, calculate_inception_score
from eval_metric import calculate_dino_similarity_score, calculate_lpips

# %%
gen_folder = "eval/ContrastiveDenoisingScore/results"
real_folder = "datatset/saved_input_images"

#%%
similarities, avg = calculate_clip_image_similarity(gen_folder, real_folder)

# %%
fid_score = calculate_fid(real_folder, gen_folder)

# %%
inception_score = calculate_inception_score(gen_folder)

# %%
calculate_lpips(gen_folder, real_folder)

# %%
calculate_dino_similarity_score(gen_folder, real_folder)

# %%
