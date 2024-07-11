# 问题   
1.只能段落内加注释，跨段落不可以    
2.对加注释的那一段信息的格式会发生改变    
3.一个文档只能加一个注释，否则会报错（可以支持多个，得修改）   
4.加批注的作者名 需要指定   


# 备注信息（加备注总结的一些信息，可能有错误）：   
注释总结：   
结构：   
word文件夹下增加三个文件：   

comments.xml   
commentsExtended.xml   
commentsIds.xml   
people.xml   
  

会修改的内容：   
word/document.xml    (把一些文字加上注释、引用上面三个文件的内容)   
word/_rels/document.xml.rels  (注意ID变化)   

后续添加新的标注：   

会修改：   
word/document.xml   
commentsExtended.xml   
[Content_Types].xml   


可能会修改：people.xml（增加新的批注人员可能就会更改，批注都是AI做的，可能只需要增加文件，后续不用需改该文件）   

总结：   
所以只需要搞清楚document.xml.rels、comments.xml、document.xml的变化规律，就可以实现批注插入的自动化。   


settings.xml  会有稍微不同   
[Content_Types].xml 会添加 添加的4个xml文件，一个xml文件2行，正好8行   
 
添加注释的时候需要修改：
word/document.xml
comments.xml
commentsExtended.xml
commentsIds.xml

# 需要用到的知识：
1.正则表达式

2.解析xml文件

3.xml.dom

4.启发：https://blog.csdn.net/FogXcG/article/details/88552355
