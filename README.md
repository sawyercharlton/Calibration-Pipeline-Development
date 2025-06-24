# Calibration-Pipeline-Development
   
## Todo
- Add bundle adjustment.

## Result Samples
- Some current result samples can be found in `asset/`

## Requirements
See `requirements.txt`

## Instructions
- Hyperparameters can be found in `src/configs/`

## Examples
 - Detect chessboard
    ```ruby
    python detect_chessboard.py
    ```
 - Calculate intrinsic parameters
    ```ruby
    python calc_intrinsic.py
    ```
 - Calculate extrinsic parameters
    ```ruby
    python calc_extrinsic.py
    ```
 - Visualize stereo detected chessboard 
    ```ruby
    python vis_chessboard.py
    ```
## Reference
- A private repo from https://github.com/sandstorm12
- https://github.com/cvlab-epfl/multiview_calib

## Acknowledgements
