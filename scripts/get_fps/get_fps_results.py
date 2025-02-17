

import os, sys, shutil, json
from pathlib import Path
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_path)
import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    counts = 0 
    time_steps = []
    size_views = len(views)
    loop_time = int(np.ceil(1000/size_views))
    for i in range(loop_time):
        for view in views:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
            # gt = view.original_image[0:3, :, :]
            # print(counts)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            time_steps.append(end_time - start_time )
    
    time_steps = np.array(time_steps)
    fps = 1 / np.mean(time_steps)
    print(f'FPS: {fps:.3f}')
    print(f'FPS without first figure: {1/np.mean(time_steps[1:]):.3f}')
    
    # data = {'FPS': f'{fps:.3f}' , 'FPS-wo-1': f'{1/np.mean(time_steps[1:]):.3f}'}
    # path_res = Path('scripts/get_fps/results.json')
    # with path_res.open(mode='w') as f:
    #     json.dump(data, f, indent=4)
    
    # fps = 
    # breakpoint()

    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)