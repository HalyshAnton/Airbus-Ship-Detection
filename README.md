# Airbus-Ship-Detection

# Data
The Airbus Ship Detection Challenge involves the task of identifying and segmenting ships in satellite images. The problem revolves around image segmentation, where the goal is to accurately delineate the boundaries of ships within a given image. This segmentation task is crucial for applications such as maritime surveillance, navigation, and environmental monitoring.

# Data
The dataset consists of satellite images paired with corresponding binary masks indicating the presence or absence of ships. The masks are encoded using the Run-Length Encoding (RLE) format. Each mask is a flattened representation of the segmented region, providing a concise description of pixel runs.

# Exploratory Data Analysis (EDA)
During the exploratory data analysis, it was observed that a significant portion of the images contains no ships. To manage computational resources efficiently and focus on relevant information, the dataset is preprocessed by cropping patches containing ships. This approach not only reduces the computational burden but also ensures that the model is trained on meaningful ship-containing regions.

# Model
The chosen model architecture for this ship detection problem is the U-Net, a convolutional neural network (CNN) widely used for image segmentation tasks. The U-Net architecture consists of a contracting path, a bottleneck, and an expansive path, allowing the model to capture both high-level features and fine details.

For training, the model is optimized using a combination of Binary Cross-Entropy (BCE) loss and Dice loss. The BCE loss ensures the model learns to predict the presence or absence of ships, while the Dice loss emphasizes the spatial agreement between predicted and ground truth masks.

# Prediction
To generate predictions, the input image is divided into overlapping patches. These patches are subjected to horizontal and vertical flips, increasing the diversity of training samples. The model is then applied to each patch, and predictions are obtained. To obtain the final segmentation masks, an averaging process is employed, considering the predictions from the original and flipped patches. This ensemble approach enhances the robustness of the predictions and improves the overall segmentation performance.
