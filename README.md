# ImageCaptioner

The streamlit deployed based app will generate the caption for the uploaded image. The model is already pretrained with VisionEncoderecoder model. This is simply called transfer learning. Using transfer learning we are simply taking the model which has been trained over a lot of convolutions, we lock that and we take this model and generate the outcome. Here in this application, the input image is converted into feature vector where features are extracted. To say by the help of Vision Encoder Decoder model, your image is encoded with numbers by the use of transformer encoder. That encoded array is fed to the model LSTM-RNN which converts the vectorial input to sequence over a set of epochs to generate captions. The input data (here it's image) pre-processed before feeding to the model. Here we are simply creating an another simple DNN (Dense Neural Network) to the pre-trained model to get the desired outcome. In the project, huggingface pretrained model has been used.

Link :
https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

Please run the below command in the terminal and in the appliaction's directory to test the application on local host:
streamlit run image_caption.py
