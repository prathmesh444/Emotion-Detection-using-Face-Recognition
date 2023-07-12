# Emotion Detection using Face Recognition
This project is created using facial expression dataset of 35000 images from kaggle->https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.
It uses haar-cascade frontal-face model for face recognition task.
  
Model 1: In this model we fine Tuned this model [MobileNet v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4) through Transfer Learning concept and used keras for facial feature extraction and creating certain feature map.
**For model-1 we have achieved 99% accuracy on train and 49% on test data which was a result of OVERFITTING**.
You can study about this model in detail here:- https://github.com/prathmesh444/Emotion-Detection-using-Face-Recognition/blob/main/Detail%20Study%20on%20Model%201.pdf

Model 2:- Furthur we improved the model's accuracy using  Validation data. Additionally We used Data Augmentation and BatchImageGenerator to increase model's performance.
This time we created out own CNN model and used OpenCV to deploy our model for realtime Emotion Detection.

**For model-2 we have achieved 69% accuracy over validation data.** 
You can study about this model in detail here:- https://github.com/prathmesh444/Emotion-Detection-using-Face-Recognition/blob/main/Detail%20Study%20on%20Model%202.pdf

Model 3:- With the release of YOLO-v8 on 2023 we decided to train out our dataset using yolo-v8 extra classification model, we used ultranlytics Library for that.

**For model-3 we managed to achieve 72.3% accuracy over validation data, which is the GLOBAL BEST ACCURACY on the current dataset.**

Model 4:- We are working on furthur improvement by incorporating AutoML to generate more optimal CNN layers with auto hyper-parameter tuning.

<img src="https://github.com/prathmesh444/Emotion-Detection-using-Face-Recognition/assets/84755719/89662fb6-153c-4a7c-98f9-1bff8419f741" alt="1" width="450" height="400"/>
<img src="https://github.com/prathmesh444/Emotion-Detection-using-Face-Recognition/assets/84755719/6241788f-deb8-4742-bf70-3bb2dbe08742" alt="1" width="450" height="400"/>

