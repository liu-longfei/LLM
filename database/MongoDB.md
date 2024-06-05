# MongoDB

## MongoDB常用命令

### cls

清屏

### show databases或者show dbs

查看当前数据库

### use 数据库名

切换到指定数据库（可以切换到没有的数据库，虽然能切进去，但不会真正创建这个数据库，因为MongoDB只有在创建数据的时候才会真正创建数据库）

## 创建集合（类似于MySQL中创建一个表格）

### db.user.insertOne( {name: "deep play"})

创建user集合，并插入一条数据

```plaintext
{
  acknowledged: true,
  insertedId: ObjectId('665ece11b1260a2d85fb3b43')
}
```

plaintext:代表是否成功

plaintext：类似于uuid

### db.users.insertMany()

插入多个数据，举个例子：[{name: "李四"},{name: "王五"}]

### db.users.find()

查找插入的内容，还可以加入字段和值，就像MySQL中的WHERE语句一样，例如：db.users.find({name:"张三"})，区分数据类型，否则可能为空

db.users.find({name:"张三"}, {name: 1, email: 1, _id: 0}),限制返回的字段，就像MySQL中的SELECT后的字段一样，将字段值为0，表示不显示

db.users.find({name:"张三"}, {email: 0}),返回除email以外的字段

#### 比较运算符

等于： 默认是等于判断， 没有运算符
⼩于：`$lt`（less than）
⼩于等于：`$lte` （less than equal）
⼤于：`$gt`（greater than）
⼤于等于：`$gte`
不等于：`$ne`

例如：db.stu.find({age:{$gte:18}})

#### 逻辑运算符

`and`：在json中写多个条件即可
查询年龄⼤于或等于18， 并且性别为true的学⽣

`db.stu.find({age:{$gte:18},gender:true})`

or:使⽤$or， 值为数组， 数组中每个元素为json
查询年龄⼤于18， 或性别为false的学⽣

`db.stu.find({$or:[{age:{$gt:18}},{gender:false}]})`
查询年龄⼤于18或性别为男⽣， 并且姓名是郭靖

`db.stu.find({$or:[{age:{$gte:18}},{gender:true}],name:'gj'})`

##### 范围运算符

使⽤"`$in`"， “`$nin`” 判断是否在某个范围内
查询年龄为18、 28的学⽣

`db.stu.find({age:{$in:[18,28]}})`

##### 支持正则表达式

使⽤`//`或`$regex`编写正则表达式
查询姓⻩的学⽣

```mongo
db.stu.find({name:/^⻩/})
db.stu.find({name:{$regex:'^⻩'}})
```

### db.users.find().limit(1)

显示查看的数量

### db.users.find().sort()

db.users.find().sort({name: 1}),对数据进行排序，name是要排序的参照字段，1表示升序，-1表示降序

### db.users.find().sort().limit(1)

db.users.find().sort({name: 1}).limit(1)，返回按照name排序最靠前的1个元素（可以按照对个字段排序，只需在字典里添加其他字段就可以）

### db.users.find().sort().limit(1).skip(1)

db.users.find().sort().limit(n).skip(m),返回按照name排序最靠前的n个元素,并跳过前m个数据，一般情况下skip和sort函数一起使用

### db.users.update()

`db.集合名称.update(<query>query ,update,{multi: boolean})`

参数query:查询条件
参数update:更新操作符
参数multi:可选， 默认是false，表示只更新找到的第⼀条记录。

```mongo
db.stu.update({name:'hr'},{name:'mnc'})   更新一条
db.stu.update({name:'hr'},{$set:{name:'hys'}})    更新一条
db.stu.update({},{$set:{gender:0}},{multi:true})   更新全部

```

updateMany()可以更新多条信息

### db.users.deleteOne()

deleteMany()可以删除多条信息

## python操作MongoDB
