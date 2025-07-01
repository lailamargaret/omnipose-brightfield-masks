import tkinter as tk
from tkinter import filedialog, Listbox, Label, Button, Scrollbar, Frame, messagebox
from threading import Thread
import os
import subprocess
#from cellpose import models
#from cellpose import io
from skimage.io import imread, imsave
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import make_multi_channels
import omnipose_brightfield_forgui
import omnipose_threaded
import process_masks
import time
import threading
import queue
import re
from glob import glob

MAX_CP_WORKERS = 8

def count_images(dir_path):
    bf_images = 0
    mask_images = 0
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith((".tif", ".tiff", ".png")):
                if "Bright Field" in filename and "masks" not in filename and "stacked" not in filename and "nuclei" not in filename and "Wells.tif" not in filename:
                    bf_images += 1
                elif "masks" in filename:
                    mask_images += 1
    return dir_path, int(bf_images/3), mask_images


def background_correction(input_dir, file_patterns=["GFP", "CY5", "RFP"]):
    """
    For each file matching one of the patterns, group images by the imaging “step”
    (i.e. the part after the fifth underscore). For example, in a filename like:
      A1_pos1_GFP_2_001_01.tif
    the step is "01".
    """
    for root, _, files in os.walk(input_dir):
        for pattern in file_patterns:
            files_grouped = {}
            for file in files:
                if (pattern in file and file.endswith((".tif", ".tiff", ".png"))
                        and "Wells" not in file and "background" not in file):
                    parts = file.split('_')
                    if len(parts) >= 6:
                        # Extract the step from the sixth part (remove any extension)
                        step = parts[5].split('.')[0]
                    else:
                        step = file.split('_')[-1].split('.')[0]
                    if step not in files_grouped:
                        files_grouped[step] = []
                    files_grouped[step].append(os.path.join(root, file))

            for step, sample_images in files_grouped.items():
                background_sum = None
                # Calculate the average background for this step group
                for filename in sample_images:
                    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                    image_blurred = cv2.GaussianBlur(image, (299, 299), 0)
                    if background_sum is None:
                        background_sum = np.zeros_like(image_blurred, dtype=np.float64)
                    background_sum += image_blurred

                background_avg = background_sum / len(sample_images)
                background_avg /= np.max(background_avg)

                background_output_dir = os.path.join(root, "background_corrected")
                os.makedirs(background_output_dir, exist_ok=True)

                # Save the background average image as a .npy file.
                # Here we include the step in the filename.
                background_filename = f"A1_pos1_{pattern}_1_001_{step}_background_avg.npy"
                background_filepath = os.path.join(background_output_dir, background_filename)
                np.save(background_filepath, background_avg)

                print(f"Background image saved: {background_filepath}")


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Cell Analysis Tool")
        self.root.geometry('1280x720')  # Set the window size
        self.selected_dirs = {}
        self.status_label = Label(self.root, text="Ready to process.")
        self.status_label.pack(pady=(10, 0), fill=tk.X)

        self.setup_ui()

        self.pipeline_paths = {
            "CY5": r"Z:\Wellslab\ToolsAndScripts\CellProfiler\pipelines\CY5_masks_from_cellpose.cppipe",
            "GFP": r"Z:\Wellslab\ToolsAndScripts\CellProfiler\pipelines\GFP_masks_from_cellpose.cppipe",
            "RFP": r"Z:\Wellslab\ToolsAndScripts\CellProfiler\pipelines\RFP_masks_from_cellpose.cppipe",
        }

    def setup_ui(self):
        frame_dirs = Frame(self.root)
        frame_dirs.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        Label(frame_dirs, text="Directories").pack()

        scrollbar = Scrollbar(frame_dirs)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_dirs = Listbox(frame_dirs, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set, width=80)
        self.listbox_dirs.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox_dirs.yview)

        Button(frame_dirs, text="Add Directories", command=self.add_directories).pack(pady=(10, 0))
        Button(frame_dirs, text="Remove Directory", command=self.remove_directory).pack(pady=(0, 10))
        Button(frame_dirs, text="Process with Cellpose", command=lambda: self.process_with_cellpose(False)).pack(
            pady=(0, 10))
        # Button(frame_dirs, text="Process with Cellpose + CellProfiler", command=self.process_with_cellpose).pack(
        #      pady=(0, 10))
        Button(frame_dirs, text="Process with CellProfiler", command=lambda: self.process_with_cellprofiler()).pack(
            pady=(0, 10))

    def add_directories(self):
        dir_path = filedialog.askdirectory(title="Select Directories", mustexist=True)
        if dir_path and dir_path not in self.selected_dirs:
            Thread(target=self.add_directory, args=(dir_path,)).start()

    def add_directory(self, dir):
        result = count_images(dir)
        self.selected_dirs[dir] = result[1:]
        self.listbox_dirs.insert(tk.END, f"{dir} (Bright Field: {result[1]}, Masks: {result[2]})")

    def populate_directories(self):
        base_path = r"Z:\Wellslab\Cytation_overflow_040723"
        if os.path.exists(base_path):
            for dir_name in os.listdir(base_path):
                full_path = os.path.join(base_path, dir_name)
                if os.path.isdir(full_path):
                    self.add_directory(full_path)
        else:
            print(f"The path {base_path} does not exist.")

    def remove_directory(self):
        selected_indices = list(self.listbox_dirs.curselection())
        for index in selected_indices[::-1]:
            dir_path = self.listbox_dirs.get(index).split(" (DAPI")[0]
            if dir_path in self.selected_dirs:
                del self.selected_dirs[dir_path]
            self.listbox_dirs.delete(index)

    def process_csv(self, directory, channel):
        """
        Combine every results_{well}.csv produced for <channel>,
        aggregate counts by Metadata_Well, and save the roll-up.
        """
        base_dir = os.path.join(directory, "cellProfiler_results", channel)
        if not os.path.isdir(base_dir):
            print(f"[WARN] No CellProfiler results for {channel} in {directory}")
            return

        # 1) gather every CSV under base_dir (well sub-folders, etc.)
        csv_paths = []
        for root, _, files in os.walk(base_dir):
            csv_paths += [os.path.join(root, f)
                          for f in files
                          if "results_" in f and not "extended" in f and not "by_well" in f]

        if not csv_paths:
            print(f"[WARN] No CSV files found for {channel} in {base_dir}")
            return

        # 2) concatenate
        df = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)

        # 3) aggregate
        agg_cols = [c for c in df.columns if c.startswith("Count_")]
        agg_funcs = {c: "sum" for c in agg_cols}
        agg_funcs["Metadata_Row"] = "first"
        agg_funcs["Metadata_Column"] = "first"

        df_agg = df.groupby("Metadata_Well").agg(agg_funcs).reset_index()

        # 4) add “bin or greater” percentages
        bin_cols = [c for c in df_agg.columns
                    if "_bin" in c and "Count_nuclei_accepted" not in c]
        for col in bin_cols:
            base = col.rsplit("_", 1)[0]
            bin_num = int(col.split("_bin")[-1])
            greater = [f"{base}_bin{i}" for i in range(bin_num, 8)]
            df_agg[f"{col}_or_greater_percent"] = (
                df_agg[greater].sum(axis=1) / df_agg["Count_nuclei_accepted"] * 100
            )

        # 5) write the roll-up next to the per-well folders
        out_path = os.path.join(base_dir, f"results_by_well_{channel}.csv")
        df_agg.to_csv(out_path, index=False)
        print(f"[OK] Aggregated CSV written → {out_path}")

    def get_well_channel_groups(self, directory):
        """
        Return a dict keyed by (well, channel) → list[(core, filename)].

        core = "well_pos_id_step"  (e.g. A01_pos1_001_02)
        channel is one of self.pipeline_paths (GFP / CY5 / RFP).
        DAPI is ignored because it is only the reference channel.
        """
        groups = {}
        for fname in os.listdir(directory):
            if not fname.lower().endswith((".tif", ".tiff", ".png")):
                continue

            parts = fname.split('_')
            if len(parts) < 6:
                continue                    # malformed name

            well, pos, channel = parts[0], parts[1], parts[2]
            if channel not in self.pipeline_paths or channel == "DAPI":
                continue                    # skip non-pipeline or DAPI files

            id_part = parts[4]
            step = parts[5].split('.')[0]   # strip extension
            core = f"{well}_{pos}_{id_part}_{step}"

            groups.setdefault((well, channel), []).append((core, fname))
        return groups

    def write_file_list(self, file_names, directory, well, channel):
        parent_directory = os.path.basename(os.path.dirname(directory))
        output_directory = os.path.join(directory, "cellProfiler_results", channel, "results_per_well")
        #output_directory = f"Z:\\Wellslab\\cellProfiler_runs\\{parent_directory}\\{os.path.basename(directory)}\\{channel}"
        os.makedirs(output_directory, exist_ok=True)
        file_list_path = os.path.join(output_directory, f"{well}_{channel}_file_list.txt")
        # file_list_path = os.path.join(output_directory, f"{channel}_file_list.txt")
        with open(file_list_path, 'w') as file_list:
            for file_name in file_names:
                file_list.write(f"{os.path.join(directory, file_name)}\n")
        return file_list_path

    def call_cellprofiler_pipeline(self, directory):
        """
        Run CellProfiler in three sequential blocks (CY5 → GFP → RFP).
        Each block launches up to MAX_CP_WORKERS parallel jobs,
        one per (well, channel) group.
        """
        # ---------- normalise the UNC path → Z: drive ------------------------
        directory = (
            directory.replace(r"//hg-fs01/research", "Z:")
                     .replace(r"//hg-fs01/Research", "Z:")
        )

        dir_files   = os.listdir(directory)
        all_groups  = self.get_well_channel_groups(directory)   # {(well,ch):[(core,fname)...]}

        # --------------------------------------------------------------------
        # helpers shared by every channel
        # --------------------------------------------------------------------
        def _csv_ready(well, channel):
            base  = os.path.join(directory, "cellProfiler_results", channel)
            fname = f"results_{well}.csv"
            return (
                os.path.isfile(os.path.join(base, fname))
                or os.path.isfile(os.path.join(base, "results_per_well", fname))
            )

        def run_single_pipeline(well, channel, files):
            """Worker that actually invokes CellProfiler for a well/channel."""
            if _csv_ready(well, channel):
                print(f"[SKIP] {well}/{channel}: results CSV already present")
                return True

            # ----------------------------------------------------------------
            # Build the list of DAPI + mask + channel files belonging to *this*
            # well / channel / time-step
            # ----------------------------------------------------------------
            valid_files = []
            for core, ch_file in files:
                well_, pos, id_part, step = core.split('_')
                dapi_base  = f"{well_}_{pos}_DAPI_1_{id_part}_{step}"
                dapi_file  = next(
                    (f for f in dir_files if f.startswith(dapi_base)
                                           and f.endswith(".tif")
                                           and "Wells.tif" not in f),
                    None
                )
                mask_file  = next(
                    (f for f in dir_files if f.startswith(dapi_base)
                                           and f.endswith("_masks.png")),
                    None
                )
                if dapi_file and mask_file:
                    valid_files += [ch_file, dapi_file, mask_file]

            if not valid_files:
                print(f"[SKIP] {well}/{channel}: no matching DAPI+mask")
                return False

            file_list = self.write_file_list(valid_files, directory, well, channel)

            cp_exe = r"C:\Program Files\CellProfiler\CellProfiler.exe"
            cmd    = [
                cp_exe, "-c", "-r",
                "-p", self.pipeline_paths[channel],
                "--file-list", file_list,
            ]
            try:
                subprocess.run(cmd, cwd=os.path.dirname(cp_exe), check=True)
                print(f"[DONE] {well}/{channel}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"[ERR]  {well}/{channel}: {e}")
                return False   # ← trigger a retry

        # --------------------------------------------------------------------
        # **MAIN LOOP** — one pass per channel
        # --------------------------------------------------------------------
        for channel in self.pipeline_paths:          # guaranteed order CY5, GFP, RFP
            channel_groups = {
                (well, ch): files
                for (well, ch), files in all_groups.items()
                if ch == channel and not _csv_ready(well, ch)
            }
            if not channel_groups:
                print(f"[INFO] Nothing to do for {channel}")
                continue

            print(f"\n=== Processing channel {channel} "
                  f"({len(channel_groups)} well(s)) ===")

            MAX_PASSES = 3
            pending    = channel_groups.copy()

            for attempt in range(1, MAX_PASSES + 1):
                if not pending:
                    break  # channel finished

                print(f"[INFO] {channel}: pass {attempt} → "
                      f"{len(pending)} remaining group(s)")

                with ThreadPoolExecutor(max_workers=MAX_CP_WORKERS) as pool:
                    futures = {
                        pool.submit(
                            run_single_pipeline, well, channel, files
                        ): (well, channel)
                        for (well, channel), files in pending.items()
                    }

                    next_pending = {}
                    for fut in as_completed(futures):
                        well, ch = futures[fut]
                        ok = False
                        try:
                            ok = fut.result()
                        except Exception as exc:
                            print(f"[ERR]  {well}/{ch}: {exc}")

                        if not ok and not _csv_ready(well, ch):
                            next_pending[(well, ch)] = channel_groups[(well, ch)]

                pending = next_pending

            # roll-up CSVs for this channel as soon as it’s done
            self.process_csv(directory, channel)
            print(f"[OK] Channel {channel} complete\n")
    
    def process_file(self, full_filepath):
        try:
            mask_filename = full_filepath.split(".ti")[0] + "_masks.png"
            # Skip if a mask file already exists
            if os.path.exists(mask_filename):
                print(f"Mask file already exists for: {full_filepath}")
                return

            print(f"Processing file: {full_filepath}")
            img = imread(full_filepath)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            masks, flows, styles = self.model.eval(img, batch_size=8, diameter=None, channels=[0, 0],
                                                   flow_threshold=0.4, cellprob_threshold=4)
            io.imsave(mask_filename, masks)
            print(f"Finished processing file: {full_filepath}")
        except Exception as e:
            print(f"Failed to process file {full_filepath}. Error: {e}")

        return full_filepath

    def process_with_cellpose(self, process_with_cellProfiler=True):
        if not self.selected_dirs:
            messagebox.showwarning("No Directories Selected", "Please add at least one directory to process.")
            return

        dirs = list(self.selected_dirs.keys())
        Thread(target=self._process_with_cellpose, args=(dirs, process_with_cellProfiler)).start()
    
    def process_with_cellprofiler(self):
            if not self.selected_dirs:
                messagebox.showwarning("No Directories Selected", "Please add at least one directory to process.")
                return

            dirs = list(self.selected_dirs.keys())
            
            for dir in dirs:
                self.call_cellprofiler_rainbow_pipeline(dir)


            #Thread(target=self._process_with_cellpose, args=(dirs, process_with_cellProfiler)).start()
    
    def call_cellprofiler_rainbow_pipeline(self, directory):
        print(f"Processing directory: {directory}")
        groups = self.get_well_groups(directory)


        
    def get_well_groups(self, directory):
        """
        Return a dict keyed by (well, channel) → list[(core, filename)].

        core = "well_pos_id"  (e.g. A01_pos1_001)
        channel is one of self.pipeline_paths (GFP / CY5 / RFP / DAPI / Bright Field).
        """
        z_by_channel = {
        "Bright Field": "Z0",
        "DAPI": "Z0",
        "GFP": "Z1",
        "RFP": "Z1",
        "CY5": "Z2"}
        groups = {}
        for fname in os.listdir(directory):
            if not fname.lower().endswith((".tif", ".tiff", ".png")):
                continue

            parts = fname.split('_')
            if len(parts) < 6:
                continue                    # malformed name

            well, channel, id_part = parts[0], parts[2], parts[4]
            step = parts[5].split('.')[0]   # strip extension

            full_pos = parts[1]
            pos, z = full_pos.split("Z")
            z = "Z" + z

            core = f"{well}_{pos}_{id_part}"
            expected_z = z_by_channel.get(channel)
            if z != expected_z:
                continue

            groups.setdefault((well, pos), []).append((channel, fname))
        
        mask_root = os.path.join(directory, "bf_merged", "model_outputs")
        if os.path.isdir(mask_root):
            for key in groups:
                well, pos = key
                pattern = os.path.join(mask_root, "*", f"{well}_{pos}_merged_*_cp_masks.tif")
                found = glob(pattern)
                if found:
                    relative_mask_path = os.path.relpath(os.path.normpath(found[0]), start=directory)
                    groups[key].append(("mask", relative_mask_path))
        return groups

    def _process_with_cellpose(self, dirs, process_with_cellProfiler=True): 
            #model_name = "reRefinedRosettes-trained-on-day-7-plate5-DAPI"
            #self.model = models.CellposeModel(gpu=True, model_type=model_name)

            total_dirs = len(dirs)
            processed_dirs = 0

            for dir in dirs:
                print(f"Processing directory: {dir}")
                self.status_label.config(text=f"Processing directory: {dir}")
                inputDir = dir
                
                result = count_images(dir)
                total_files = result[2]
                # image_files = []
                    # for root, _, filenames in os.walk(dir):
                    #     image_files.extend([os.path.join(root, f) for f in filenames if
                    #                         f.endswith((".tif", ".tiff"))
                    #                         and "Bright Field" in f and "masks" not in f and "Wells.tif" not in f])
                    # total_files = len(image_files)
                    # processed_files = 0

                # for image_file in image_files:
                #     self.process_file(image_file)
                #     processed_files += 1
                #     status_text = f"Processing: {dir}\nDirectory: {processed_dirs + 1}/{total_dirs}\nFiles: {processed_files}/{total_files}"
                #     self.status_label.config(text=status_text)
                #     self.root.update()
                #     print(f"Processed {processed_files}/{total_files} files in directory: {dir}")

                # processed_dirs += 1

                try:
                    print(f"Starting {inputDir}")
                    output_dir = os.path.join(inputDir, "bf_merged")
                    os.makedirs(output_dir, exist_ok=True)
                    #print(f'Merged images being saved at {output_dir}')

                    max_queue_size = 1
                    dir_queue = queue.Queue(maxsize=max_queue_size)
                    lock = threading.Lock()

                    print(f'Starting consumer thread')
                    consumer_thread = threading.Thread(target=omnipose_brightfield_forgui.consumer, args=(output_dir, lock, dir_queue))
                    consumer_thread.start()

                    print(f'Starting producer')
                    subdir = os.path.basename(inputDir)
                    base_input = os.path.dirname(inputDir)
                    omnipose_brightfield_forgui.producer(base_input, subdir, output_dir, dir_queue)

                    dir_queue.put(None)
                    consumer_thread.join()

                except Exception as e:
                    print(f'Failed {inputDir}: {e}')
            
            processed_dirs += 1
            self.status_label.config(text="Processing complete.")
            messagebox.showinfo("Processing Complete", "Cellpose processing is complete.")
            print("Cellpose processing is complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
