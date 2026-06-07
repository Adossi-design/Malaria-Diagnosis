# Automated Malaria Diagnosis Using CNN and Transfer Learning

This project compares five convolutional neural network models for classifying parasitized and uninfected blood cells from microscopy images. Each model went through seven experiments to find what settings and architectures work best for this task.

**Group 6 | Formative 2 | Introduction to Machine Learning | African Leadership University**

<br>

## Team

| Member | Model | Best Experiment | F1 | AUC |
|--------|-------|-----------------|----|-----|
| Adossi Fred William | Baseline CNN | Exp 5 (Dropout 0.5) | 0.9647 | 0.9938 |
| Nehemie Ishimwe | Advanced CNN | Exp 5 (GAP + BatchNorm) | 0.9647 | 0.9927 |
| Berissa Muyizere | VGG16 | Exp 4 (Unfreeze block3+4+5) | 0.9648 | 0.9931 |
| Elvin Cyubahiro | ResNet50 | Exp 4 (Unfreeze 50 layers) | 0.9623 | 0.9920 |
| Gabriel Sezibera Tuyisingize | MobileNetV2 | Exp 4 (Unfreeze 40 layers) | 0.9626 | 0.9923 |

<br>

## Dataset

We used the [NIH Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-dataviewer.html) from the National Library of Medicine which has 27,558 single cell microscopy images taken from Giemsa stained blood smear slides. The dataset is perfectly balanced with 13,779 parasitized cells that contain visible Plasmodium ring structures and 13,779 uninfected cells that look like normal red blood cells. All images were resized to 128x128 pixels and we split them 80/20 using a fixed seed of 42 so every model trains and tests on the exact same data giving us 22,046 training images and 5,512 test images.

<br>

## Repository Structure

```
Malaria-Diagnosis/
├── Malaria-Diagnosis_CNN_Group11-EvenNumbers.ipynb   # combined notebook with all five models
└── models/
    ├── Adossi_BaselineCNN.ipynb                      # simple 2 block CNN trained from scratch
    ├── Nehemie_AdvancedCNN.ipynb                     # deeper CNN with BatchNorm and GAP
    ├── Berissa_VGG16.ipynb                           # VGG16 with progressive block unfreezing
    ├── Elvin_ResNet50.ipynb                          # ResNet50 with frozen BatchNorm layers
    └── Gabriel_MobileNetV2.ipynb                     # MobileNetV2 lightweight transfer learning
```

<br>

## The Models

### Baseline CNN (Adossi)

This is the simplest model in the group and it was built to give everyone a performance floor to compare against. It has two convolutional blocks where each block runs a Conv2D layer then MaxPooling2D and the output feeds into a Dense classifier head. The seven experiments changed one thing at a time including learning rate, dropout at 0.3 and 0.5, data augmentation, and batch size so we could see exactly what each setting does to performance.

### Advanced CNN (Nehemie)

This model goes deeper than the baseline using three convolutional blocks with 32, 64 and 128 filters and adds BatchNormalization after each conv layer to keep training stable. It also replaces the Flatten layer with GlobalAveragePooling2D which cuts the number of parameters in the classifier head from about 32,000 down to 128. The seven experiments tested how each of these additions changes performance by removing them one at a time and also tried L2 regularization and switching from Adam to SGD with Nesterov momentum.

### VGG16 (Berissa)

VGG16 was pretrained on ImageNet and already knows how to detect edges, textures and shapes so Berissa replaced only the output layer with a GlobalAveragePooling2D layer, a Dense layer with 256 units, Dropout at 0.5 and a sigmoid output for binary classification. The seven experiments started with the backbone fully frozen and then gradually unfroze more convolutional blocks using a very small learning rate of 0.00001 to avoid erasing what the model learned from ImageNet. Two experiments also tested data augmentation and a wider 512 unit head.

### ResNet50 (Elvin)

ResNet50 uses skip connections that let gradients flow through all 50 layers without disappearing which is what makes training a network this deep actually work. The same classification head structure as VGG16 was used but one important difference is that all BatchNormalization layers stayed frozen throughout every experiment because they store statistics from 1.2 million ImageNet images and recalibrating them on our small dataset would break the normalization and destabilize training. The seven experiments tested unfreezing 0, 10, 30 and 50 layers plus augmentation and a very low learning rate of 5e-6.

### MobileNetV2 (Gabriel)

MobileNetV2 uses depthwise separable convolutions and inverted residual blocks to get good accuracy with far fewer parameters than VGG16 or ResNet50 which makes it the most practical model for deployment in a clinic with limited computing resources. The classification head uses Dropout at 0.4 and BatchNorm layers are kept frozen just like in ResNet50. The seven experiments followed the same progressive unfreezing strategy testing 0, 10, 20 and 40 unfrozen layers and also tested augmentation and a wider 512 unit head.

<br>

## Shared Pipeline

All five models use the exact same data pipeline so the comparison is fair. These are the shared functions that every notebook relies on:

| Function | What it does |
|----------|-------------|
| `create_splits()` | Creates the 80/20 train and test folders with seed 42 |
| `make_generators()` | Loads images with model specific preprocessing and optional augmentation |
| `evaluate_model()` | Returns accuracy, precision, recall, F1, AUC, confusion matrix and ROC curve |
| `get_callbacks()` | Sets up EarlyStopping with patience 5, ModelCheckpoint and ReduceLROnPlateau |
| `train_or_load()` | Saves model weights to Google Drive and skips retraining if they already exist |

When augmentation is turned on the generator applies random horizontal and vertical flips, rotations up to 20 degrees, zoom up to 15% and width and height shifts up to 10%.

<br>

## Results

All five models were tested on the same 5,512 image test set and the numbers below show the best result each model achieved across its seven experiments.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| VGG16 (Berissa) | 0.9643 | 0.9516 | 0.9782 | **0.9648** | 0.9931 |
| Advanced CNN (Nehemie) | 0.9643 | 0.9529 | 0.9768 | 0.9647 | 0.9927 |
| Baseline CNN (Adossi) | 0.9650 | 0.9717 | 0.9579 | 0.9647 | 0.9938 |
| MobileNetV2 (Gabriel) | 0.9623 | 0.9547 | 0.9706 | 0.9626 | 0.9923 |
| ResNet50 (Elvin) | 0.9621 | 0.9560 | 0.9688 | 0.9623 | 0.9920 |

Something interesting about these results is that all five models ended up within 0.003 F1 of each other which was not what we expected going in. The pretrained models did not have as much of an advantage over the from scratch models as the literature usually suggests and one reason for that is the 128x128 input size we used which is smaller than the 224x224 that VGG16 and ResNet50 were originally designed for.

One finding that was consistent across all three pretrained models is that data augmentation only helped when enough layers were unfrozen to let the backbone actually respond to the harder training images. When we added augmentation to a frozen backbone it made performance worse every time because the classification head alone does not have enough capacity to handle the added difficulty.

<br>

## How to Run

The notebooks run on **Google Colab** with GPU enabled.

1. Click the Colab badge at the top of any notebook to open it
2. Mount your Google Drive when the notebook asks so that trained weights can be saved and reloaded across sessions
3. Set `downloadData = True` on your first run to download the NIH dataset
4. Run all cells from top to bottom and the `train_or_load()` function will automatically skip any experiment that already has saved weights

If you want to rerun a specific experiment just set `rebuild = True` in that cell or delete the saved weights from your Drive folder.

<br>

## References

- Rajaraman et al. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection. *PeerJ, 6*, e4568.
- He et al. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.
- Sandler et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*, 4510–4520.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*, 448–456.
- Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering, 22*(10), 1345–1359.
- Tajbakhsh et al. (2016). Convolutional neural networks for medical image analysis: Full training or fine tuning? *IEEE Transactions on Medical Imaging, 35*(5), 1299–1312.
- World Health Organization. (2023). *World Malaria Report 2023*.
