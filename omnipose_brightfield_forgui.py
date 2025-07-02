import os
import make_multi_channels
import omnipose_threaded
import process_masks
import time
import threading
import queue

def producer(inputDir, subdir, outputDir, queue, threads=4):
    subdir = make_multi_channels.make_multi_channels_reorder(inputDir, subdir, outputDir, threads)
    queue.put(subdir)

def producer_worker(inputDir, outputDir, dir_queue, threads=4):
    subdirs = [d for d in os.listdir(inputDir) if os.path.isdir(os.path.join(inputDir, d)) and not "epoch" in d and not "bf_merged" in d]
    
    for subdir in subdirs:
        while dir_queue.full():
            time.sleep(0.25)

        producer(inputDir, subdir, outputDir, dir_queue, threads)
    
    dir_queue.put(None)

def consumer(outputDir, lock, queue, threads=4):
    while True:
        subdir = queue.get()
        if subdir is None:
            break

        merged_img_dir = outputDir
        
        dead_dir = None
        for model_info_array in PRETRAINED_MODEL_INFOS:
            try:
                live_dir = omnipose_threaded.run_omnipose(merged_img_dir, model_info_array, save_tif=True, save_flows=False, save_outlines=False, num_threads=threads)
                results_csv = process_masks.process_mask_files(live_dir, CM_PER_PIXEL, PLATE_AREAS.get(PLATE_TYPE), force_save=False, filter_min_size=None)
            except Exception as e:
                print(f"Error processing directory {merged_img_dir} with {model_info_array}: {e}")
        if DEAD_MODEL_INFO:
            try:
                dead_dir = omnipose_threaded.run_omnipose(merged_img_dir, DEAD_MODEL_INFO[0], save_tif=True, save_flows=True, save_outlines=False, num_threads=threads)
                results_csv = process_masks.process_mask_files(dead_dir, CM_PER_PIXEL, PLATE_AREAS.get(PLATE_TYPE), force_save=False, filter_min_size=None)
            except Exception as e:
                print(f"Error processing directory {merged_img_dir} with dead model: {e}")

        queue.task_done()

PLATE_TYPE = "96W"
MAGNIFICATION = "10x"
CYTATION = True
PLATE_AREAS = {"6W": 9.6, "12W": 3.8, "24W": 2, "48W": 1.1, "96W": 0.32}
CM_PER_MICRON = 1 / 10000
if CYTATION:
    MICRONS_PER_PIXEL = 1389 / 1992 if MAGNIFICATION == "10x" else 694 / 1992
    IMAGE_AREA_CM = 1992 * 1992 * MICRONS_PER_PIXEL**2 * CM_PER_MICRON**2
else:
    if MAGNIFICATION=="10x":
        MICRONS_PER_PIXEL = 0.61922571983322461  # Taken from image metadata EVOS 10x
        IMAGE_AREA_CM = 0.0120619953 # Taken from image metadata EVOS 10x
    else:
        MICRONS_PER_PIXEL = 1.5188172690164046  # Taken from image metadata EVOS 4x
        IMAGE_AREA_CM = 0.0725658405  # Taken from image metadata EVOS 4x

CM_PER_PIXEL = CM_PER_MICRON*MICRONS_PER_PIXEL


PRETRAINED_MODEL_INFOS = [
    #10x NPC
    [r"Z:\Wellslab\ToolsAndScripts\OmniposeBrightfieldMasks\cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_NEW-MERGED-TRAIN-FROM-DAPI_2024_03_29_02_03_10.875324_epoch_960", 0.4, 0],
    ]

DEAD_MODEL_INFO = None

# WIP
# DEAD_MODEL_INFO = [[r"E:\MERGED_sspsygene_growthassay_colonies\ded\models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_2_nchan_3_dim_2_ded_2025_03_07_02_10_15.252341_epoch_3999", 0.4, 1]]

def oldmain():
    inputDirs = [r"Z:\Wellslab\Cytation_overflow_040723\LS_TEST"]

    for inputDir in inputDirs:
        try:
            # output_dir = os.path.join(inputDir, "bf_merged")
            # print(f'Merged images being saved at {output_dir}')
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir, exist_ok=True)
            
            print(f"starting {inputDir}")
            base_dir = r"Z:\Wellslab\Cytation_overflow_040723\LS_TEST\merged_"
            output_dir = make_multi_channels.create_merged_directory(inputDir, base_dir)
            print(f'Merged images being saved at {output_dir}')

            max_queue_size=999
            dir_queue = queue.Queue(maxsize=max_queue_size)
            lock = threading.Lock()

            print(f'Starting consumer threads')
            consumer_threads = []  # List to keep track of threads
            for _ in range(1):  # Create and start consumer threads
                consumer_thread = threading.Thread(target=consumer, args=(output_dir, lock, dir_queue))
                consumer_thread.start()
                consumer_threads.append(consumer_thread)

            print(f'Starting producer threads')
            producer_thread = threading.Thread(target=producer_worker, args=(inputDir, output_dir, dir_queue))
            producer_thread.start()
            producer_thread.join()

            for _ in range(len(consumer_threads)):
                dir_queue.put(None)

            for consumer_thread in consumer_threads:
                consumer_thread.join()

        except:
            print(f'failed {inputDir}')
def main():
    inputDirs = [r"Z:\Wellslab\Cytation_overflow_040723\LS_TEST\day1plate1read1"]  # <- specific subdir only

    for inputDir in inputDirs:
        try:
            print(f"Starting {inputDir}")
            output_dir = os.path.join(inputDir, "bf_merged")
            os.makedirs(output_dir, exist_ok=True)
            #print(f'Merged images being saved at {output_dir}')

            max_queue_size = 1
            dir_queue = queue.Queue(maxsize=max_queue_size)
            lock = threading.Lock()

            print(f'Starting consumer thread')
            consumer_thread = threading.Thread(target=consumer, args=(output_dir, lock, dir_queue))
            consumer_thread.start()

            print(f'Starting producer')
            subdir = os.path.basename(inputDir)
            base_input = os.path.dirname(inputDir)
            producer(base_input, subdir, output_dir, dir_queue)

            dir_queue.put(None)
            consumer_thread.join()

        except Exception as e:
            print(f'Failed {inputDir}: {e}')

if __name__ == "__main__":
    main()