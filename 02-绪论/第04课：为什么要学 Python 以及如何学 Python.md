# 为什么要学习python？

* 自己从事java和scala开发有一段时间了，期间也用过python，在学习ai的过程中，对python的强大深有体会
  * 比如：python中的切片思想，是其他语言无法比拟的
  * python 是一名动态语言，开发简单，第三方包功能强大
  * python做数据分析，numpy、scipy、pandas、matplotlib这些模块大大简化了数据分析的难度
  * python做爬虫也是其他语言无法比拟的
  * python调用sklearn，tensorflow，keras的包封装的很完善简介，而用java就不太好。

* scala vs python vs java
  * scala主要用于开发spark，sparkmllib主要是做大数据下的机器学习，在这方面，scala+sparkmllib是很有优势的
  * scala 和 java 都是运行在jvm中，满足于性能的高要求，适合于高并发，大数据量的业务
  * scala是一种函数式编程，类似于python
  * java是一种静态语言，在开发大型web项目，或者复杂的业务逻辑时适合用java，尤其是在高并发的业务场景下
  * python训练好的模型，一般保存为pb格式，对外提供服务时，还是要用java对加载模型，毕竟使用rpc调用还是java方便，python使用http接口也可以，但是内网调用不方便
  * python的优势在于数据展示，数据分析，jupyter notebook做图表数据展示时更能体现它的强大，python作为AI的主要开发语言，未来前景一片光明
  * 我给自己的定位是工程能力+算法能力，把技术定位为解决业务的工具，还不是局限于技术本身。