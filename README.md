# Airbus-Ship-Detection

The Airbus Ship Detection Challenge involves the task of identifying and segmenting ships in satellite images. The problem revolves around semantic image segmentation, where the goal is to accurately delineate the boundaries of ships within a given image. This segmentation task is crucial for applications such as maritime surveillance, navigation, and environmental monitoring.

# Data
The dataset consists of satellite images paired with corresponding binary masks indicating the presence or absence of ships. The masks are encoded using the Run-Length Encoding (RLE) format. Each mask is a flattened representation of the segmented region, providing a concise description of pixel runs.

For downloading data run next line ! kaggle competitions download -c 'airbus-ship-detection'

# Exploratory Data Analysis (EDA)
During the exploratory data analysis, it was observed that a significant portion of the images contains no ships. 

![alt text](https://github.com/HalyshAnton/Airbus-Ship-Detection/blob/main/images/ship_count_distribution.png)

Also size of images is too big so for managing computational resources efficiently imeges was croped into patches and only those ones with ships were selected(or random patch if original image doesn't contain ships). This approach not only reduces the computational burden but also ensures that the model is trained on meaningful ship-containing regions.

![alt text](https://github.com/HalyshAnton/Airbus-Ship-Detection/blob/main/images/cropping.png)

After all about 10 000 images was selected.

# Model
The chosen model architecture for this ship detection problem is the U-Net with pretrained EddicientNetB0 encoder.

For training, the model is optimized using a combination of Binary Cross-Entropy (BCE) loss and Dice loss. This combination leads to more stable training with unbalaced prediction masks(ships 

# Prediction
To generate predictions, the input image is divided into overlapping patches. These patches are subjected to horizontal and vertical flips. The model is then applied to each patch, and average prediction is obtained. Then predictions from patches are merged to final result. This ensemble approach called TTA enhances the robustness of the predictions and improves the overall segmentation performance.

![alt text](https://github.com/HalyshAnton/Airbus-Ship-Detection/blob/main/images/prediction.png)

# Inference
For training load directory with training images you should unzip data/images.zip and run line py train.py
For testing run line py test.py "path_to_image"
