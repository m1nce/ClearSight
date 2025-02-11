# ClearSight
 
A dynamic adaptation framework for real-time object detection in adverse environmental conditions, such as fog and glare. This project combines a lightweight condition classifier with specialized object detection sub-models, enabling efficient and accurate detection in dynamically changing weather conditions.

To see our proposal, refer to [reports/proposal.pdf](https://github.com/m1nce/ClearSight/blob/main/reports/proposal.pdf).

<!-- SETUP -->
## Setup:
1. Ensure that Conda installed. If it isn't, you can download [Miniconda](https://docs.anaconda.com/miniconda/)
   or [Anaconda](https://docs.anaconda.com/anaconda/install/) and install it.

2. Clone the git repository.
```sh
git pull https://github.com/m1nce/ClearSight.git
```

3. Move to the repository directory and create conda environment.
```sh
conda env create -f environment.yml
```

4. Activate the environment (this must be done everytime your terminal is closed!).
```sh
conda activate clearsight
```

5. Register for a Cityscapes account and password. This is necessary to download the data 
   used in this repository. This is done in [this link](https://www.cityscapes-dataset.com/register/).

6. Create an `.env` file in repository and enter in your username and password in the following format:
```
export USERNAME=your_username
export PASSWORD=your_password
```

7. Download the data.
```sh
cd utils
chmod +x get_data.sh
./get_data.sh
```

8. Augment the data to include glaring and foggy conditions.
```sh
python augment_cityscapes.py
```

<!-- CONTRIBUTORS -->
## Created by:
* [Minchan Kim](https://github.com/m1nce)
* [Andy Ho](https://github.com/handy0102)
