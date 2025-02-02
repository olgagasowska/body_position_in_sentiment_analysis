## body_position_in_sentiment_analysis
This project explores **multimodal sentiment analysis** using the **MELD dataset**, incorporating both text and video features. The goal is to improve emotion classification accuracy by integrating **pose estimation** (head movement, posture, and head tilt angles) extracted using MediaPipe. The project compares the Text - Only Model and Text + Video Model in sentiment prediction. 


## **Dataset used:**
**DATASET MELD** https://github.com/declare-lab/MELD The model bases only on the train part of the data due to time restriction of the project

Data preparation: The models were trained on a small sample (around 3560 elements) so the accuracy of the results may be biased.

The dataset was used on the basis of the preprocessed data (https://github.com/declare-lab/MELD/tree/master/data/MELD_Dyadic), and the raw files provided by MELD creators (mp4).

## **Visual Features extracted via MediaPipe:**
| **Feature**               | **Description** |
|---------------------------|----------------|
| **Raw Positions (x, y, z)** | Absolute positions of key landmarks (e.g., nose, shoulders) |
| **Head Tilt Angle (Roll, Pitch, Yaw)** | Orientation of the head relative to the body |
| **Head Movement Magnitude** | Overall displacement of the head over a period of time |
| **Posture Deviation** | Difference in shoulder height (asymmetry) |

The usage of these features focuses on one group of body features: motionless (Head Movement Magnitude seen as the change in position, but not as a characteristic of movement). The goal of the model is to establish if the characteristics of body position improve the sentiment prediction in sentiment analysis.



Steps undertaken:

    Importing preprocessed data(“Utterance”, “Speaker”, “Sentiment”) and raw files.

    Creating a “Filename” for each entrance in the preprocessed data that allows to align the video data with the rest of the data

    Exporting head and posture features from the raw files with MediaPipe, with average for each “Utterance”

    Aligning the two datasets. Dataset structure: “Filename”, “Avg Head Movement (x,y)”, “Avg Posture”, “Speaker”, “Sentiment”, “Utterance”


Two models were created:

    Text - Only where the model establishes the sentiment on the basis of “utterance” only

    Text + Video where the model combines the text and video layer for each utterance.


## **RESULTS:**
## Model Performance Comparison  

This table compares the performance of the **Text-Only Model** and the **Text + Video Model** based on key evaluation metrics.  

| **Metric**       | **Text-Only Model** | **Text + Video Model** |
|------------------|--------------------|------------------------|
| **Accuracy**     | 68.28%             | 76.21%                 |
| **F1 Score**     | 68.23%             | 75.92%                 |
| **Precision**    | Negative: 0.62  <br> Neutral: 0.78  <br> Positive: 0.57  | Negative: 0.78  <br> Neutral: 0.78  <br> Positive: 0.71  |
| **Recall**       | Negative: 0.58  <br> Neutral: 0.79  <br> Positive: 0.60  | Negative: 0.65  <br> Neutral: 0.87  <br> Positive: 0.70  |
| **Average Loss** | Epoch 1: 0.91  <br> Epoch 4: 0.32  | Epoch 1: 4.54  <br> Epoch 6: 0.89  |

The **Text + Video Model** outperforms the **Text-Only Model** across all key metrics, demonstrating the benefit of incorporating video-based features into sentiment analysis.  


## Per-Speaker Accuracy Comparison  

This table compares the accuracy of the **Text-Only Model** and the **Text + Video Model** for each speaker. The **Text + Video Model** generally improves performance across most speakers.  

| **Speaker**           | **Text-Only Model** | **Text + Video Model** |
|-----------------------|--------------------|------------------------|
| Barry                | 1.000000           | 0.642857               |
| Bernice              | 1.000000           | 1.000000               |
| Bobby                | 1.000000           | 0.333333               |
| Both                 | 1.000000           | 1.000000               |
| Mike                 | 1.000000           | 1.000000               |
| Gunther              | 1.000000           | N/A                    |
| Drunken Gambler      | 1.000000           | 1.000000               |
| Marc                 | 1.000000           | 0.000000               |
| Raymond              | 1.000000           | 1.000000               |
| Robert               | 1.000000           | 1.000000               |
| Mrs. Green           | 1.000000           | 1.000000               |
| Mona                 | 0.833333           | 1.000000               |
| Tag                  | 0.764706           | 0.647059               |
| Dr. Miller           | 0.750000           | 0.750000               |
| Stage Director       | 0.750000           | 0.500000               |
| Monica               | 0.723577           | 0.739130               |
| Joey                 | 0.721311           | 0.801205               |
| Dana                 | 0.714286           | 0.428571               |
| Phoebe               | 0.695238           | 0.810606               |
| Ross                 | 0.670588           | 0.752137               |
| Pete                 | 0.666667           | 1.000000               |
| Julie                | 0.666667           | 0.800000               |
| Rachel               | 0.662500           | 0.742081               |
| Chandler             | 0.604396           | 0.745614               |
| The Fireman          | 0.600000           | 0.400000               |
| All                  | 0.500000           | 1.000000               |
| Nurse                | 0.500000           | 0.666667               |
| Charlie              | 0.500000           | 1.000000               |
| The Casting Director | 0.500000           | 1.000000               |
| Receptionist         | 0.500000           | 0.400000               |
| Kristin              | 0.333333           | 0.666667               |
| Dr. Green            | 0.000000           | 1.000000               |
| Mr. Tribbiani        | 0.000000           | 1.000000               |
| Mrs. Geller          | 0.000000           | 1.000000               |
| Ross and Joey        | 0.000000           | 1.000000               |
| Wayne                | 0.000000           | 1.000000               |

The **Text + Video Model** improves accuracy for most speakers, particularly those who had lower accuracy in the **Text-Only Model**. However, a few speakers saw a decrease in performance.  

## **Conclusions and Concerns**

Although it seems that the Text + Video model performs significantly better in the sentiment prediction, it must be taken into account the the multimodal model may not only predict the sentiment, but may learn the characteristics of a particular person. It can identify the person basing on the visual features, and consequently, label a certain person to a specific sentiment.

The Speakers who appear in both models with 100% accuracy have a small variance and tend to express one Sentiment prevalently. Also, the number of these occurrences are less or equal to 7 only. This may lead to unreliable results.  

The fact that among six Speakers with the highest number of occurrences in the dataset (see folder Images), there is also a visible variability in the sentiment labels, may indicate that the Model identifies certain visual cues and uses them to predict the sentiment. However, it may be also connected to the fact that the Model learns to interpret their unique body language and it may not be as powerful when encountering a new person. **In order to analyze the nature of this, a further study with more naturalistic recordings and a greater speaker variability should be conducted.**




