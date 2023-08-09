# Transformer_Limu 
# 一、简述
这是一个对《动手学深度学习》pdf中Transformer代码块进行注释的实战，修复了该篇章中直接运行提供的代码报错的问题，并对代码给出了详细的注释。对于可以运行的部分，也用[have a try]进行了标注。

#二、注意事项
想必你已经安装好了torch等包，下面讲一下我在运行中遇到的困难。
1.修改torch.py文件
torch.py文件中，需要在read_data_nmt()函数中，添加“encoding=utf-8”，但是一直报错，如下图所示，无法保存：
![image](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/75af5945-2752-4c86-9b54-8a58b2da2c92)
原因是在对torch.py文件进行修改的时候会弹出一个窗口，我们需要选择第三项（开始我就是选的第一项qwq）
![image](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/72ad35a5-8322-4f7b-a46d-dcd073c571cb)
然后就可以进行修改了。

2.文中的代码
该pdf中给出的代码多是使用类与对象，所以想要深刻地理解代码，就必须把类与类的继承了解清楚。其中的几个难点是父类函数、super()函数、__init__()函数。
在笔者的Transformer.py文件中，我用[have a try]注释进行了分块，在运行时，只需要将[have a try]及其下面的代码取消注释，方可运行。

3.一些快捷键分享
注释、取消注释：ctrl+/
打出间隔：选中一部分代码，tab
取消间隔：shift+tab
