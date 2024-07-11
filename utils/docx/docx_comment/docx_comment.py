import os
import re
import shutil
from docx import Document
from xml.dom.minidom import parseString
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

from utils import comments_xml, commentsIds_xml, commentsExtended_xml, people_xml
from utils import content_types, built_comment_text


ET.register_namespace('wpc', "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas")
ET.register_namespace('cx', "http://schemas.microsoft.com/office/drawing/2014/chartex")
ET.register_namespace('cx1', "http://schemas.microsoft.com/office/drawing/2015/9/8/chartex")
ET.register_namespace('cx2', "http://schemas.microsoft.com/office/drawing/2015/10/21/chartex")
ET.register_namespace('cx3', "http://schemas.microsoft.com/office/drawing/2016/5/9/chartex")
ET.register_namespace('cx4', "http://schemas.microsoft.com/office/drawing/2016/5/10/chartex")
ET.register_namespace('cx5', "http://schemas.microsoft.com/office/drawing/2016/5/11/chartex")
ET.register_namespace('cx6', "http://schemas.microsoft.com/office/drawing/2016/5/12/chartex")
ET.register_namespace('cx7', "http://schemas.microsoft.com/office/drawing/2016/5/13/chartex")
ET.register_namespace('cx8', "http://schemas.microsoft.com/office/drawing/2016/5/14/chartex")
ET.register_namespace('mc', "http://schemas.openxmlformats.org/markup-compatibility/2006")
ET.register_namespace('aink', "http://schemas.microsoft.com/office/drawing/2016/ink")
ET.register_namespace('am3d', "http://schemas.microsoft.com/office/drawing/2017/model3d")
ET.register_namespace('o', "urn:schemas-microsoft-com:office:office")
ET.register_namespace('r', "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
ET.register_namespace('m', "http://schemas.openxmlformats.org/officeDocument/2006/math")
ET.register_namespace('v', "urn:schemas-microsoft-com:vml")
ET.register_namespace('wp14', "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing")
ET.register_namespace('wp', "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing")
ET.register_namespace('w10', "urn:schemas-microsoft-com:office:word")
ET.register_namespace('w', "http://schemas.openxmlformats.org/wordprocessingml/2006/main")
ET.register_namespace('w14', "http://schemas.microsoft.com/office/word/2010/wordml")
ET.register_namespace('w15', "http://schemas.microsoft.com/office/word/2012/wordml")
ET.register_namespace('w16cid', "http://schemas.microsoft.com/office/word/2016/wordml/cid")
ET.register_namespace('w16se', "http://schemas.microsoft.com/office/word/2015/wordml/symex")
ET.register_namespace('wpg', "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup")
ET.register_namespace('wpi', "http://schemas.microsoft.com/office/word/2010/wordprocessingInk")
ET.register_namespace('wne', "http://schemas.microsoft.com/office/word/2006/wordml")
ET.register_namespace('wps', "http://schemas.microsoft.com/office/word/2010/wordprocessingShape")
ET.register_namespace('Ignorable', "w14 w15 w16se w16cid wp14")


# 新建文件夹，并将文件解压缩到指定的文件夹
def check_file_path(docx_path, target_path):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)

    doc = ZipFile(docx_path)
    doc.extractall(path=target_path)


# 创建comments.xml, commentsIds.xml, commentsExtended.xml, people.xml
def create_comments_commentIds_commentsExtended(target_path, comments_xml, commentsIds_xml, commentsExtended_xml, people_xml):
    with open(os.path.join(target_path, 'word/comments.xml'), 'w', encoding='utf-8') as f:
        f.write(comments_xml)
    with open(os.path.join(target_path, 'word/commentsIds.xml'), 'w', encoding='utf-8') as f:
        f.write(commentsIds_xml)
    with open(os.path.join(target_path, 'word/commentsExtended.xml'), 'w', encoding='utf-8') as f:
        f.write(commentsExtended_xml)
    with open(os.path.join(target_path, 'word/people.xml'), 'w', encoding='utf-8') as f:
        f.write(people_xml)


# 修改word/_rels/document.xml.rels文件内容
def modify_document_xml_rels(target_path):
    document_xml_rels_path = os.path.join(target_path, 'word/_rels/document.xml.rels')
    ET.register_namespace('', "http://schemas.openxmlformats.org/package/2006/relationships")
    tree = ET.parse(document_xml_rels_path)
    root = tree.getroot()
    # 遍历Relationship的属性，找到最大的Id
    id_list = []
    for sub in root:
        id_list.append(sub.attrib.get("Id"))
    if len(id_list) > 0:
        id_list = sorted(id_list)
    id_max = int(id_list[-1].replace('rId', ''))
    # 为document.xml.rels文件添加新的内容
    comments_element = ET.Element("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    comments_element.attrib = {'Id': f"rId{id_max+1}",
                               'Type': "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments",
                               "Target": "comments.xml"}
    commentsExtended_element = ET.Element("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    commentsExtended_element.attrib = {'Id': f"rId{id_max + 2}",
                                       'Type': "http://schemas.microsoft.com/office/2011/relationships/commentsExtended",
                                       "Target": "commentsExtended.xml"}
    commentsIds_element = ET.Element("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    commentsIds_element.attrib = {'Id': f"rId{id_max + 3}",
                                  'Type': "http://schemas.microsoft.com/office/2016/09/relationships/commentsIds",
                                  "Target": "commentsIds.xml"}
    people_element = ET.Element("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    people_element.attrib = {'Id': f"rId{id_max + 4}",
                             'Type': "http://schemas.microsoft.com/office/2011/relationships/people",
                             "Target": "people.xml"}
    root.append(comments_element)
    root.append(commentsExtended_element)
    root.append(commentsIds_element)
    root.append(people_element)

    new_xml_str = ET.tostring(root, 'utf-8')
    parsed = parseString(new_xml_str)
    pretty_xml_as_string = parsed.toprettyxml()
    # pretty_xml_as_string = parsed.toprettyxml(indent="  ")
    pretty_xml_as_string = pretty_xml_as_string.replace('<?xml version="1.0" ?>',
                                                        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')

    with open(document_xml_rels_path, 'w') as f:
        f.write(pretty_xml_as_string)


# 修改 [content_Types].xml文件内容
def modify_content_types_xml(target_path):
    content_types_xml_path = os.path.join(target_path, '[content_Types].xml')
    ET.register_namespace('', "http://schemas.openxmlformats.org/package/2006/content-types")
    tree = ET.parse(content_types_xml_path)
    root = tree.getroot()
    new_xml_str = ET.tostring(root, 'utf-8')
    parsed = parseString(new_xml_str)
    pretty_xml_as_string = parsed.toprettyxml()
    pretty_xml_as_string = pretty_xml_as_string.replace('<?xml version="1.0" ?>',
                                                        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
    pretty_xml_as_string = pretty_xml_as_string.replace('</Types>', content_types)
    print(pretty_xml_as_string)
    with open(content_types_xml_path, 'w') as f:
        f.write(pretty_xml_as_string)


# 修改word/document.xml文件内容-->根据给定内容加批注
def modify_word_document(docx_path, target_path, comment_content):
    word_document_xml_path = os.path.join(target_path, 'word/document.xml')
    # 查询含有命名空间的字符串,为后期替换不带命名空间的<w:document>做准备
    with open(word_document_xml_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    pattern = r'<w:document.*?>'
    xml_document_str = re.search(pattern, xml_str, re.DOTALL).group()
    # print(xml_document_str)

    # 确定加批注的内容在第几段落
    doc = Document(docx_path)
    idx_para = -1
    for idx, paragraph in enumerate(doc.paragraphs):
        # print(idx, paragraph.text)
        if comment_content in paragraph.text:
            # print(paragraph.text)
            idx_para = idx
            break
    if idx_para == -1:
        return None

    # 解析xml文件
    tree = ET.parse(word_document_xml_path)
    root = tree.getroot()

    # 为xml段落和document的段落寻找起增量，便于根据document的段落定位到xml的段落
    para_idx_add = [0]  # document的段落和xml段落的增量列表
    for idx, para in enumerate(root[0]):
        if idx == 0:
            continue
        elif idx != 0 and para.tag.endswith('}p'):
            para_idx_add.append(para_idx_add[idx - 1])
        else:
            para_idx_add.append(para_idx_add[idx - 1] + 1)

    # 在xml中的段落
    xml_para_num = idx_para + para_idx_add[idx_para]

    # 构建批注的段落
    comment_para_element_text = built_comment_text(comment_content, doc.paragraphs[idx_para].text)

    # 获取xml段落的paraId
    for key, val in root[0][xml_para_num].attrib.items():
        if 'paraId' in key:
            paraId = val
    print(paraId)

    new_xml_str = ET.tostring(root)
    xml_parsed = parseString(new_xml_str)
    pretty_xml_str = xml_parsed.toprettyxml()

    # 找到需要替换掉的comment段落
    pattern = f'<w:p.*?"{paraId}".*?">(.*?)</w:p>'
    old_comment_str = re.search(pattern, pretty_xml_str, re.DOTALL).group(1)

    # 替换内容
    pretty_xml_str = pretty_xml_str.replace(old_comment_str, comment_para_element_text)
    pretty_xml_str = pretty_xml_str.replace('<?xml version="1.0" ?>',
                                            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
    print(pretty_xml_str)

    with open(os.path.join(target_path, 'word/document.xml'), 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)


# 将解压的文件重新压缩成docx文件
def zip_files(zip_name, folder_path):
    # 创建一个ZipFile对象，并设置模式为写
    with ZipFile(zip_name, 'w', ZIP_DEFLATED) as zipf:
        # os.walk()可以遍历文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 创建压缩文件中的文件路径
                file_path = os.path.join(root, file)
                # 将文件添加到zip文件中
                zipf.write(file_path, os.path.relpath(file_path, folder_path))


if __name__ == '__main__':
    docx_path = r"C:\Users\LW0016778\Desktop\test_docx\test\temp1.docx"
    target_path = r"C:\Users\LW0016778\Desktop\test_docx\test\unzip"
    new_docx_file = r"C:\Users\LW0016778\Desktop\test_docx\test\temp1-test.docx"
    comment_content = '识别表格中的内'

    check_file_path(docx_path, target_path)
    # 创建comments.xml, commentsIds.xml, commentsExtended.xml, people.xml
    create_comments_commentIds_commentsExtended(target_path, comments_xml, commentsIds_xml, commentsExtended_xml, people_xml)
    # 修改word/_rels/document.xml.rels文件内容
    modify_document_xml_rels(target_path)
    # 修改 [content_Types].xml文件内容
    modify_content_types_xml(target_path)
    # 修改word/document.xml文件内容-->根据给定内容加批注
    modify_word_document(docx_path, target_path, comment_content)

    # 将解压的文件重新压缩成docx文件
    zip_files(new_docx_file, target_path)
