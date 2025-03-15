# ClearSight

A dynamic adaptation framework for real-time object detection in adverse environmental conditions, such as fog and glare. This project combines a lightweight condition classifier with specialized object detection sub-models, enabling efficient and accurate detection in dynamically changing weather conditions.

<p align="center">
  <img src="img/overview_fig.png" alt="drawing" width="600"/>  
</p>

To view the project's motivation, refer to [reports/proposal.pdf](https://github.com/m1nce/ClearSight/blob/main/reports/proposal.pdf).

<!-- SETUP -->
## Setup:
1. Ensure that Conda installed. If it isn't, you can download [Miniconda](https://docs.anaconda.com/miniconda/)
   or [Anaconda](https://docs.anaconda.com/anaconda/install/) and install it. If you are on Windows, please install it to 
   [WSL](https://learn.microsoft.com/en-us/windows/wsl/install). [This GitHub Gist](https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da) has a very good tutorial to do so. 

2. Clone the git repository.
```sh
git pull https://github.com/m1nce/ClearSight.git
```

3. Move to the repository directory and create a new conda environment.
```sh
cd ClearSight
conda env create -f environment.yml
```

4. Activate the environment (this must be done everytime your terminal is closed!).
```sh
conda activate clearsight
```

5. Download the tiny Cityscapes and gtfine data.
```sh
cd utils
chmod +x tiny_data.sh
chmod +x tiny_gtfine.sh
bash tiny_data.sh
bash tiny_gtfine.sh
```

6. Augment the Cityscapes data to include glaring and foggy conditions.
```sh
python augment_cityscapes.py
```

7. Train the MobileNet model.
```sh 
python train_model.py
```

<!-- CONTRIBUTORS -->
## Created by:
* [Minchan Kim](https://github.com/m1nce)
* [Andy Ho](https://github.com/handy0102)
