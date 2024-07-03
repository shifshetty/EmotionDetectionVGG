# Human Emotion and Facial Feature Recognition 

The aim of this project was to build a deep learning model that is capabale of classifying the emotion (positive/negative/surprise) depicted in an input image along with identifying the presence of few facial features such as raised eyebrows, dimples, lip pressor etc. The model built will be a multi-output network with two tasks: <br/>
1) Multi class classification for Emotion recognition 
2) Multi label classification for FACS codes identification 

Our dataset, the Cohn-Kanade Dataset, contains labeled images of faces depicting various emotions from over 123 subjects, totaling 560 images. A provided CSV file lists image paths, emotion classes, and FACs categories. We have 3 emotion classes and 15 FACs categories. Each image belongs to one emotion class but can have multiple FACs codes.                              
Using a base VGG16 model with transfer learning, the final layers were added with the VGG16 trainable parameter set to False, ensuring pre-trained weights remain unchanged. A global average pooling layer was first added to reduce the dimensions of the feature maps. Two Dense layers with 512 neurons each, using ReLU activation, were included to capture complex patterns while preventing overfitting. Dropout layers with a 0.3 rate were inserted after each Dense layer.
Two output branches were defined: the first for emotion classification with 3 neurons using a softmax activation, and the second for multi-label FACS code classification with 15 nodes using a sigmoid activation. This setup allows for accurate classification of emotions and the presence of multiple FACS codes in the images.
