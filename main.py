from torchvision import  transforms
import streamlit as st
from PIL import Image
from io import BytesIO
from model import ImageClassifier
import torch
import pathlib
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT) 

test_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE)])



@st.cache_resource
def load_model():
    model = ImageClassifier()
    # model.load_state_dict(torch.load('model.pth'))
    model = torch.load(
    pathlib.Path('./model.pth').as_posix(),  
    map_location=lambda storage, loc: storage
)
    model.eval()   
    return model


def check_img(uploaded_file):
    if not uploaded_file:
        raise TypeError('File is missing')

    if not isinstance(uploaded_file, BytesIO):
        raise TypeError('File wrong format')

    try:
        img = test_transform(Image.open(uploaded_file))
    except Exception as err:
        print(err)
        raise TypeError('File is not image')

    return img


def model_pred(img, model):
    predict = model(img)[0]
    return predict['generated_text']


def load_image(model):
    uploaded_file = st.file_uploader(
        label='Выберите изображение для описания')

    try:
        img = check_img(uploaded_file)
    except Exception as err:
        st.write(str(err))
        return

    st.image(img)

    result = st.button('Распознать изображение')
    if result:
        predict = model_pred(img, model)
        st.write(
            f'**Результаты распознавания: {predict}**')


if __name__ == '__main__':

    st.title('Описание изображений онлайн')
    model = load_model()
    load_image(model)