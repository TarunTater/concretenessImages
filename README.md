# Imageability estimation framework

## Dependencies

- Python 3.6.x
- Sklearn 0.19.x
- OpenCV 3.4.x (with -contrib, if using FeatureType.SURF)
- Progressbar2 3.x
- For YOLO_* features: https://github.com/madhawav/YOLO3-4-Py
- For GIST feature: https://github.com/tuttieee/lear-gist-python

# How to use

## Predict values for new words

To run the pretrained model on your own imagesets to predict the imageability of new words, you want to run `preprocess.py` on your own imagesets to extract visual information, and then use `predict.py` to predict values using the model.

The model will expect n images for each word. The same number of images for each word is needed and set by the pretrained model (n=1000, 2500, 5000 available in `/models/`.)

## Train own models

If you need to train model for other feature combinations or number of input images, use `generate_models.py` (Or ask @mkasu.) You need the dataset which is available in `/work/kastnerm/imageability_flickr` or can be re-crawled using `crawler.py`.

__Note__: The cache files for all pretrained models is available in `/work/kastnerm/deepimageability/cache/` and `/work/kastnerm/deepimageability/eigenvalues/`. Copy them to `./cache/` and `./eigenvalues/` to vastly improve the generation of new models.

## All scripts

`preprocess.py`
- run to preprocess visual features for each word
- will take a long time
- you should preprocess certain features on CPU/GPU to reduce time needed (see below)
- preprocessing will cache to `./cache/`
- delete ./cache/ to redo calculation for certain words

`predict.py`
- run to predict values for words with pretrained model
- only run after preprocessing the visual features for all words/features. otherwise this will take forever...

`generate_models.py`
- use to generate a new model trained with the imageability dataset crawled from Flickr
- takes multiple weeks to run if not in cache
- distribute across multiple servers!

`crawler.py`
- generate an imageability dataset crawled from YFCC100M dataset
- dataset file structure is described in Section "Imageability flickr dataset"

`evaluate.py`
- generate do experiments and evaluations used for [3]

`cleanup.py`
- clean up a dataset by deleting duplicates, corrupt images, and GIFs
- it will look in all subfolders from a given root, expecting the format described in "Dataset format" below
- you can call it like `./crawler.py /work/kastnerm/test/`

`lib/*.py`
- a mess of Helper libraries used from various previous projects.
- not everything might be runnable in its current state because of legacy code.

# Additional comments

## Pretrained models

I pretrained Random Forest models for 1000, 2500, or 5000 input images, respectively. Each model uses the following features: `[FeatureType.YOLO_NUM_9000, FeatureType.YOLO_COMPOSITION, FeatureType.SURF, FeatureType.ColorHSV, FeatureType.GIST]`.

NOTE: If the input dataset is not from YFCC100M, we need to skip `FeatureType.VisualConcepts` (H1 from [3]) because the used annotations are inherited from the Flickr dataset and thus not available for random images.

## Feature selection

- You probably want to run YOLO_* on a GPU machine (e.g. konXX)
   - All of them need pydarknet
- You probably want to run SURF/GIST on a high-CPU machine (e.g. sakuraXX)
   - GIST needs lear-gist-python which does not have a pip package (compile by hand on sakura.)
   - SURF uses a pre-trained Bag-of-Visual Words model which is pretrained using approx. 3 mio images from YFCC100M.

You should precalculate the cache for all visual features per server in seperate and then use cached values to calculate predictions on local machine with pretrained model.

## Dataset format

`predict.py`/`preprocess.py` expect the following format:

`PREFIX/word1/*.jpg`  
`PREFIX/word2/*.jpg`  
...

e.g.:  
`/work/kastnerm/test/adventure/1328901231.jpg`, `.../3429130128.jpg` etc.  
`/work/kastnerm/test/time/3178290371.jpg`, `.../3478101210.jpg` etc.  
with `PREFIX=/work/kastnerm/test/`

The script needs the same amount of images for each word. E.g. 5000 images for each word. It will use the first 5000 images it finds if there are more than 5000 images.

Make sure all images can be processed by OpenCV - corrupted image files might crash the script. Use `cleanup.py` to delete all noisy images (deleting duplicates, corrupt images, and GIFs.)

You can define the prefix and num_images in the header of each main python script. 

For dataset access there are two helper functions in `lib/create_matrix.py`:  
`def processLooseWord(word, location, features, num_images, returnEigenvalues=True):`  
`def processImageabilityDatasetTerm(term, features, num_images, returnEigenvalues=True):`

The first will expect a data format as described above, the second will provide an API for the imageability flickr dataset used in [2] and [3].

### Imageability flickr dataset

The crawled Flickr dataset uses a special data format. 

The images files are located in `/work/kastnerm/imageability_flickr/`. Each image has a hashed name which also specifies its location by the first 3 letters in its name. 
E.g. `865b8d498d1fb82e92a7e808e82c4111.jpg` is placed in the folder `/imageability_flickr/865/865b8d498d1fb82e92a7e808e82c4111.jpg`.

There is a JSON lookup table in `/work/kastnerm/imageability_flickr-cur.json` which can be used to find which images correspond to which words.

One image can be attached to multiple words.

The function `processImageabilityDatasetTerm()` will create an `Imageset` object from the input words with `num_images` number of images sampled from the lookup table.

# Publications

[1] Estimating the visual variety of concepts by referring to Web popularity. Marc A. Kastner, Ichiro Ide, Yasutomo Kawanishi, Takatsugu Hirayama, Daisuke Deguchi, Hiroshi Murase. Multimed Tools Appl, 78(7), 9463-9488, August 2018. doi: 10.1007/s11042-018-6528-x.

[2] A preliminary study on estimating word imageability labels using Web image data mining. Marc A. Kastner, Ichiro Ide, Yasutomo Kawanishi, Takatsugu Hirayama, Daisuke Deguchi, Hiroshi Murase. 言語処理学会第25回年次大会, March 2019.

[3] Estimating the imageability of words by mining visual characteristics from crawled image data. Marc A. Kastner, Ichiro Ide, Frank Nack, Yasutomo Kawanishi, Takatsugu Hirayama, Daisuke Deguchi, Hiroshi Murase. Multimed Tools Appl, Published online, 32p, February 2020. doi: 10.1007/s11042-019-08571-4.
