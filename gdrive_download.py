import gdown

# File ID from Google Drive link
file_id = '1J2wykqkGdRe5ttM1OJf_TsiPD5JO-qlb'

# Construct the download URL
url = f'https://drive.google.com/uc?id={file_id}'

# Output file path
output = '/workspaces/Image-Classification-using-CNN-Mobile-Phones-and-Laptops/mobile_laptop_model/200model.keras'

# Download the file
gdown.download(url, output, quiet=False)
