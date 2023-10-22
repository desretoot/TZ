from torchvision import  transforms
import streamlit as st
from PIL import Image
from io import BytesIO
from model import ImageClassifier
import torch
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT) 

test_transform = transforms.Compose([transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE)])


@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = ImageClassifier().to(device)
    state = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.to(torch.device('cpu'))

    model.eval()   
    return model


def check_img(uploaded_file):
    if not uploaded_file:
        raise TypeError('File is missing')

    if not isinstance(uploaded_file, BytesIO):
        raise TypeError('File wrong format')

    try:
        img = Image.open(uploaded_file)
    except Exception as err:
        print(err)
        raise TypeError('File is not image')

    return img


def model_pred(img, model):
    img_trans = test_transform(img)
    predict = model(img_trans.unsqueeze(0))[0]
    if torch.argmax(predict) == 0:
        res = 'кот :cat:'
    else:
        res = 'собака :dog:'
    return res

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
        st.markdown(f'### Результаты распознавания: {predict}')


if __name__ == '__main__':

    st.title('Распознавание котиков и собачег')
    model = load_model()
    load_image(model)