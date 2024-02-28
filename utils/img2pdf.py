# jpg转为pdf  链接：https://zhuanlan.zhihu.com/p/664658396
import img2pdf

images = [r"D:\data\paper\a180978fc2337febe001ab17001a6a27373596439—2.jpg"]
output = "animals.pdf"

# 创建一个PDF文件 并以二进制方式写入
with open(output, "wb") as f:
    # convert函数 用来转PDF
    write_content = img2pdf.convert(images)
    f.write(write_content)  # 写入文件
print("转换成功！")  # 提示语
