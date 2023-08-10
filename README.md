# 一、简述
这是一个对《动手学深度学习》pdf中Transformer代码块进行注释的实战，修复了该篇章中直接运行提供的代码报错的问题，并对代码给出了详细的注释。对于可以运行的部分，也用[have a try] 进行了标注。

pdf下载：https://zh-v2.d2l.ai/d2l-zh-pytorch.pdf

# 二、注意事项

想必你已经安装好了torch等包，下面讲一下我在运行中遇到的困难。

## 1.修改torch.py文件

在运行代码块的时候，你应该会遇到一个与d2l.load_nmt()函数有关的问题，就像这个博主描述的这样：https://blog.csdn.net/m0_48085801/article/details/126611608

根据他的提醒，torch.py文件中，需要在read_data_nmt()函数中，添加“encoding=utf-8”，但是一直报错，如下图所示，无法保存：

![image](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/75af5945-2752-4c86-9b54-8a58b2da2c92)

原因是在对torch.py文件进行修改的时候会弹出一个窗口，我们需要选择第三项（开始我就是选的第一项qwq）

![image](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/72ad35a5-8322-4f7b-a46d-dcd073c571cb)

然后就可以进行修改了。

## 2.文中的代码

该pdf中给出的代码多是使用类与对象，所以想要深刻地理解代码，就必须把类与类的继承了解清楚。其中的几个难点是父类函数、super()函数、\_\_init\_\_()函数。
在笔者的Transformer.py文件中，我用[have a try]注释进行了分块，在运行时，只需要将[have a try]及其下面的代码取消注释，方可运行。

## 3.对于Transformer的理解
代码的部分，难点在于Encoderblock与Decoderblock的输入分别是什么，笔者使用processon绘制了用函数形式的流程图，方便理解记忆。同时，提供两张CSDN上面的图片供大家参考。
特别是Decoderblock块在训练阶段与测试阶段的不同。下面是图片以及参考的CSDN文章。

![image](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/0ce1ee79-a505-49e0-adb3-5e0048199f4d)
![Decoderblock](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/2ac637ef-aadb-4709-8fa4-90e1a1ab4bd2)
![Train_process](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/ee396240-8621-465b-866b-f16f1805416e)
![Test_process](https://github.com/jfbbcom/Transformer_Limu/assets/106417483/680e9e31-a7d1-497a-bb46-71a61356f21c)



## 3.一些快捷键分享

- 注释、取消注释：ctrl+/

- 打出间隔：选中一部分代码，tab

- 取消间隔：shift+tab
