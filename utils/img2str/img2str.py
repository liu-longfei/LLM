import base64

with open('img_12.png', 'rb') as image_file:
    encoded_string = base64.b64encode(image_file.read())

image_string = encoded_string.decode('utf-8')
print(image_string)

# 假设image_string包含了Base64编码的图像字符串
decoded_image = base64.b64decode(image_string)

with open('img_12_t.jpg', 'wb') as file:
    file.write(decoded_image)
