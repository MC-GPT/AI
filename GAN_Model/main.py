from util import *
import torchvision.utils as vutils
import boto3

from crop import crop_image
from projector import image_inversion
from toonify import img_toonify

def s3_connection():
    try:
        # s3 클라이언트 생성
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id="{액세스 키 ID}",
            aws_secret_access_key="{비밀 액세스 키}",
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!") 
        return s3

img_name = 'Cat'

crop_image('/Users/hoon/Desktop/AI/GAN_Model/imgs/{}.jpg'.format(img_name),'/Users/hoon/Desktop/AI/GAN_Model/imgs')

image_inversion(f'/Users/hoon/Desktop/AI/GAN_Model/imgs/crop_{img_name}.jpg','/Users/hoon/Desktop/AI/GAN_Model/inversion_codes')

real_latent = torch.load(f'/Users/hoon/Desktop/AI/GAN_Model/inversion_codes/{img_name}.pt',map_location='mps')['latent']
real_latent = style2list(real_latent)

# Assuming `img_toonify` returns the toonified image
img_toonify(real_latent,f'/Users/hoon/Desktop/AI/GAN_Model/imgs/toonify_{img_name}.jpg')


s3 = s3_connection()

try:
    s3.upload_file(f'/Users/hoon/Desktop/AI/GAN_Model/imgs/toonify_{img_name}.jpg','MC_NUGU',f'toonify_{img_name}.jpg')
except Exception as e:
    print(e)