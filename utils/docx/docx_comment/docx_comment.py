import os
import shutil
from xml.dom.minidom import parseString
from zipfile import ZipFile, ZIP_DEFLATED
from xml.etree import ElementTree as ET

from utils import comments_xml, commentsIds_xml, commentsExtended_xml, people_xml
from utils import content_types


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
def modify_word_document(target_path, comment_content):
    word_document_path = os.path.join(target_path, '/word/document.xml')


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

    check_file_path(docx_path, target_path)
    # 创建comments.xml, commentsIds.xml, commentsExtended.xml, people.xml
    create_comments_commentIds_commentsExtended(target_path, comments_xml, commentsIds_xml, commentsExtended_xml, people_xml)
    # 修改word/_rels/document.xml.rels文件内容
    modify_document_xml_rels(target_path)
    # 修改 [content_Types].xml文件内容
    modify_content_types_xml(target_path)
    # 修改word/document.xml文件内容-->根据给定内容加批注
    modify_word_document(target_path)

    # 将解压的文件重新压缩成docx文件
    # zip_files(new_docx_file, target_path)
