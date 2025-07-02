import os
import subprocess
import random
import os
import random
import time
import subprocess
import concurrent.futures
import shutil

def run_on_subset(images_subset, model_info_array, savedir, save_tif, save_flows, save_outlines):
    time.sleep(random.uniform(0, 10))
    tempdir = f"Z:/Wellslab/ToolsAndScripts/OmniposeBrightfieldMasks/tempdir/temp_{random.randint(0, 99999)}_{random.randint(0, 99999)}"
    savedir = os.path.normpath(savedir)
    savedir = savedir.replace("\\", "/")
    print(f"run subset called {savedir}")


    os.makedirs(tempdir, exist_ok=True)
    for image in images_subset:
        symlink_path = os.path.join(tempdir, os.path.basename(image))
        try:
            os.symlink(image, symlink_path)
        except Exception as e:
            try:
                shutil.copy2(image, symlink_path)
            except Exception as copy_err:
                print(f"Failed to create symlink and failed to copy {image}. Error: {copy_err}")

    cmd_base = [
        "omnipose",
        "--no_npy",
        "--no_suppress",
        "--affinity_seg",
        "--nclasses", "2",
        "--flow_threshold", f"{model_info_array[1]}",
        "--mask_threshold", f"{model_info_array[2]}",
        "--dir", tempdir,
        "--pretrained_model", model_info_array[0],
        "--savedir", savedir,
        "--all_channels",
        "--nchan", model_info_array[0].split("nchan_")[1][0], "--verbose"
    ]

    if save_tif:
        cmd_base.append("--save_tif")
    if save_flows:
        cmd_base.append("--save_flows")
    if save_outlines:
        cmd_base.append("--save_outlines")

    cmd = ["C:\\Users\\Wells\\miniconda3\\Scripts\\conda.exe", "run", "-n", "omnipose", "-v"] + cmd_base
    #print(f'Running omnipose command for {cmd}')
    
    subprocess.run(cmd, capture_output=True, text=True)
    


def run_omnipose(directory_path, model_info_array, save_tif, save_flows, save_outlines, num_threads=4):
    print(f"Starting omnipose proccessing in: {directory_path}")

    savedir = f"{directory_path}/model_outputs/"+"_".join(model_info_array[0].split("_")[-8:])+f"_{model_info_array[1]}_{model_info_array[2]}"
    os.makedirs(savedir, exist_ok=True)

    existing_mask_files = {f.replace("_cp_masks.tif", "") for f in os.listdir(savedir) if '_cp_masks.tif' in f}
    all_images_with_root = [os.path.join(directory_path, img) for img in os.listdir(directory_path)
                            if (img.endswith('.tif') or img.endswith('.tiff') or img.endswith('.TIF')) 
                            and not any(suffix in img for suffix in ['masks', 'dP.tif', 'outlines.tif', 'flows.tif', 'Wells', 'cellProb'])
                            and os.path.basename(img).replace(".tif", "") not in existing_mask_files]

    random.shuffle(all_images_with_root)
    if len(all_images_with_root) != 0:
        image_subsets = []
        if len(all_images_with_root) > 1:
            num_splits = min(num_threads, len(all_images_with_root))
            images_per_split = len(all_images_with_root) // num_splits
            remainder = len(all_images_with_root) % num_splits

            start = 0
            for i in range(num_splits):
                end = start + images_per_split + (1 if i < remainder else 0)
                image_subsets.append(all_images_with_root[start:end])
                start = end
        else:
            num_splits=1
            image_subsets = [all_images_with_root]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_splits) as executor:
            executor.map(run_on_subset, image_subsets, [model_info_array]*len(image_subsets), 
            [savedir]*len(image_subsets), [save_tif]*len(image_subsets), [save_flows]*len(image_subsets), [save_outlines]*len(image_subsets))

    return savedir