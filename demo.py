# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np
import cv2

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# if DEFAULT_DEVICE == "mps":
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def select_points(frame):
    """
    Allow user to select points on the first frame.
    Returns: List of (x,y) coordinates
    """
    points = []
    window_name = "Select tracking points (press SPACE when done, ESC to cancel)"
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw a small circle at clicked point
            cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow(window_name, display_frame)
    
    # Create a copy of the frame for display
    display_frame = frame.copy()
    cv2.imshow(window_name, display_frame)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE key to finish
            cv2.destroyAllWindows()
            return points
        elif key == 27:  # ESC key to cancel
            cv2.destroyAllWindows()
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default=None,
        help="path to a video",
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        help="path to a segmentation mask",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/scaled_offline.pth",
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive point selection mode",
    )

    args = parser.parse_args()

    # Check if video path is provided
    if args.video_path is None:
        raise ValueError("Please provide a video path using --video_path")

    # Check if video file exists
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    # load the input video frame by frame with error handling
    try:
        video = read_video_from_path(args.video_path)
        if video is None:
            raise ValueError(f"Failed to load video from {args.video_path}")
        
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    except Exception as e:
        print(f"Error loading video: {str(e)}")
        raise

    if args.mask_path:
        segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
        segm_mask = torch.from_numpy(segm_mask)[None, None]
    else:
        segm_mask = None

    model = CoTrackerPredictor(
        checkpoint=args.checkpoint,
        window_len=60,
    )

    model = model.to(DEFAULT_DEVICE)
    video = video.to(DEFAULT_DEVICE)

    # Convert first frame to numpy for OpenCV
    first_frame = video[0, 0].permute(1, 2, 0).cpu().numpy()
    first_frame = (first_frame * 255).astype(np.uint8)

    # Get points from user
    selected_points = select_points(first_frame)

    if selected_points is None or len(selected_points) == 0:
        print("No points selected. Exiting.")
        exit()

    # Convert points to queries format (t, x, y)
    queries = torch.zeros((1, len(selected_points), 3), device=DEFAULT_DEVICE)
    for i, (x, y) in enumerate(selected_points):
        queries[0, i] = torch.tensor([0, x, y])

    # Run tracking with selected points
    chunk_size = 50  # Process 50 frames at a time to avoid out of memory
    all_tracks = []
    all_visibility = []
    
    for i in range(0, video.shape[1], chunk_size):
        video_chunk = video[:, i:i + chunk_size]
        
        if i == 0:
            # For the first chunk, use original queries
            chunk_queries = queries
        else:
            # For subsequent chunks, use the last known positions as starting points
            last_tracks = all_tracks[-1]
            last_positions = last_tracks[:, -1]  # Get positions from last frame of previous chunk
            chunk_queries = torch.zeros_like(queries)
            chunk_queries[:, :, 0] = 0  # Set time index to 0 for new chunk
            chunk_queries[:, :, 1:] = last_positions  # Use last known positions
        
        pred_tracks, pred_visibility = model(
            video_chunk,
            backward_tracking=args.backward_tracking,
            segm_mask=segm_mask,
            queries=chunk_queries
        )
        
        # Store results
        all_tracks.append(pred_tracks)
        all_visibility.append(pred_visibility)
    
    # Concatenate results along time dimension
    pred_tracks = torch.cat(all_tracks, dim=1)
    pred_visibility = torch.cat(all_visibility, dim=1)

    print("computed")

    # save a video with predicted tracks
    seq_name = args.video_path.split("/")[-1]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3, mode='optical_flow', tracks_leave_trace=50)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
    )
