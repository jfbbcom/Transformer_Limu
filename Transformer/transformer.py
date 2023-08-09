import math
import pandas as pd
import os
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 10.7.2 基于位置的前馈⽹络
# 基于位置的前馈⽹络对序列中的所有位置的表⽰进⾏变换时使⽤的是同⼀个多层感知机（MLP），这就是称
# 前馈⽹络是基于位置的（positionwise）的原因。在下⾯的实现中，输⼊X的形状（批量⼤⼩，时间步数
# 或序列⻓度，隐单元数或特征维度）将被⼀个两层的感知机转换成形状为（批量⼤⼩，时间步数，
# ffn_num_outputs）的输出张量。
class PositionWiseFFN(nn.Module):
    # 定义基于位置的前馈网络模型
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
    **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 初始化方法，定义模型的参数和层
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)# 第一个线性层
        self.relu = nn.ReLU()                                   # ReLU激活函数
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)# 第二个线性层
    # 前向传播方法
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))# 线性变换 -> ReLU -> 线性变换
# # [have a try]
# # 下⾯的例⼦显⽰，改变张量的最⾥层维度的尺⼨，会改变成基于位置的前馈⽹络的输出尺⼨。因为⽤同⼀个
# # 多层感知机对所有位置上的输⼊进⾏变换，所以当所有这些位置的输⼊相同时，它们的输出也是相同的。
# ffn = PositionWiseFFN(4, 4, 8)
# ffn.eval()
# print(ffn(torch.ones((2, 3, 4)))[0])

# 10.7.3 残差连接和层规范化
# # [have a try]
# # 以下代码对.不同维度的层规范化和批量规范化的效果。
# # 创建 LayerNorm 和 BatchNorm1d 实例
# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
# # 创建输入张量 X
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# # 在训练模式下计算 X 的均值和方差，并应用 LayerNorm 和 BatchNorm1d
# layer_norm_output = ln(X)  # 应用 LayerNorm
# batch_norm_output = bn(X)  # 应用 BatchNorm1d
# # 打印结果
# print('layer norm:', layer_norm_output, '\nbatch norm:', batch_norm_output)
class AddNorm(nn.Module):
# """残差连接后进⾏层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        #类的构造函数 __init__()，接收 normalized_shape 和 dropout 等参数
        super(AddNorm, self).__init__(**kwargs)
        #调用父类的构造函数来初始化该类的实例
        self.dropout = nn.Dropout(dropout)
        #创建一个 dropout 层对象，其中 dropout 变量是传入的 dropout 参数
        self.ln = nn.LayerNorm(normalized_shape)
        #创建一个 LayerNorm（层归一化）层对象，其中 normalized_shape 是传入的规范化形状参数
    def forward(self, X, Y):#前向传播函数 forward()，接收输入张量 X 和 Y 作为参数
        return self.ln(self.dropout(Y) + X)
# # [have a try]
# # 传入 [3, 4] 作为规范化形状，0.5 作为 dropout 参数，并赋值给变量 add_norm
# add_norm = AddNorm([3, 4], 0.5)
# #评估模式，停止计算梯度与dropout
# add_norm.eval()
# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

#10.7.4 编码器Encoder
# 有了组成Transformer编码器的基础组件，现在可以先实现编码器中的⼀个层。下⾯的EncoderBlock类
# 包含两个⼦层：多头⾃注意⼒和基于位置的前馈⽹络，这两个⼦层都使⽤了残差连接和紧随的层规范化。
class EncoderBlock(nn.Module):
    # """Transformer编码器块"""
    # key_size：多头注意力层中键（key）的维度大小。
    # query_size：多头注意力层中查询（query）的维度大小。
    # value_size：多头注意力层中值（value）的维度大小。
    # num_hiddens：隐藏层的维度大小。用于词嵌入，表示每个词或符号将被编码成一个多长的向量表示。
    # norm_shape：层归一化操作（AddNorm）中的归一化维度大小。
    # ffn_num_input：前向传播层输入的维度大小。
    # ffn_num_hiddens：前向传播层隐藏层的维度大小。
    # num_heads：多头注意力层中的头数。
    # dropout：Dropout层的丢弃率。
    # use_bias：是否使用偏置项。
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        # 多头注意力层
        self.attention = d2l.MultiHeadAttention(
                    key_size, query_size, value_size, num_hiddens, num_heads,
                    dropout,use_bias)
        #残差链接与层归一化1
        self.addnorm1 = AddNorm(norm_shape, dropout)
        #前向传播层
        self.ffn = PositionWiseFFN(
                ffn_num_input, ffn_num_hiddens, num_hiddens)
        # 残差链接与层归一化2
        self.addnorm2 = AddNorm(norm_shape, dropout)
    # Transformer 编码器块的前向传播过程forward
    # self:等价于实例对象，即一个编码器模块
    # X:输入（K、Q、V）
    # valid_lens：输入的有效长度
    # Y: 经过残差连接和层归一化后的中间结果
    # Z: X经过一个encoder模块的结果
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        Z = self.addnorm2(Y, self.ffn(Y))
        return Z
# # [have a try]
# #正如从代码中所看到的，Transformer编码器中的任何层都不会改变其输⼊的形状。
# X = torch.ones((2, 100, 24))
# valid_lens = torch.tensor([3, 2])
# encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
# encoder_blk.eval()
# print(encoder_blk(X, valid_lens).shape)

# 下⾯实现的Transformer编码器的代码中，堆叠了num_layers个EncoderBlock类的实例。由于这⾥
# 使⽤的是值范围在-1和1之间的固定位置编码，因此通过学习得到的输⼊的嵌⼊表⽰的值需要先乘以嵌⼊维
# 度的平⽅根进⾏重新缩放，然后再与位置编码相加。
# num_layers个encoder模块的编码器
class TransformerEncoder(d2l.Encoder):
    # vocab_size：词汇表大小。
    # key_size：多头注意力层中键（key）的维度大小。
    # query_size：多头注意力层中查询（query）的维度大小。
    # value_size：多头注意力层中值（value）的维度大小。
    # num_hiddens：隐藏层的维度大小。表示每个词或符号将被编码成一个多长的向量表示。
    # norm_shape：层归一化操作（AddNorm）中的归一化维度大小。
    # ffn_num_input：前向传播层输入的维度大小。
    # ffn_num_hiddens：前向传播层隐藏层的维度大小。
    # num_heads：多头注意力层中的头数。
    # num_layers：编码器块的数量。
    # dropout：Dropout层的丢弃率。
    # use_bias：是否使用偏置项。
    # **kwargs：任意关键参数
    def __init__(self, vocab_size, key_size, query_size, value_size,
                num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        # 词向量的大小
        self.num_hiddens = num_hiddens
        # embedding将输入的离散索引序列转换为密集的向量表示。
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 词向量传给位置编码,通过正弦和余弦生成一组固定的位置编码向量,Dropout随机丢弃位置编码
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 顺序容器，用于存放多个编码块
        self.blks = nn.Sequential()
        # 对于编号为0到num_layers-1的encoder代码块，将其依次添加进self实例对象
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                    norm_shape, ffn_num_input, ffn_num_hiddens,
                    num_heads, dropout, use_bias))
    # 定义一个 TransformerEncoder 类的前向传播过程
    # X：输入数据张量，形状为(batch_size, seq_length, embedding_dim)。
    # valid_lens：有效长度张量，用于掩盖填充部分，形状为(batch_size, )。
    def forward(self, X, valid_lens, *args):
        # 词嵌入(索引->向量)+缩放(向量与位置编码的数值范围匹配)+添加位置信息。
        # 结果传给X(张量)。与原来的X相比，数值改变，形状一致。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 存储每个编码块的注意力权重。列表的长度为编码块的数量，使用None初始化。
        self.attention_weights = [None] * len(self.blks)
        # 遍历，获取块的索引 i 和块对象 blk
        for i, blk in enumerate(self.blks):
            # 调用 EncoderBlock 类的 forward 方法，得到编码结果，并将结果赋值给变量 X
            X = blk(X, valid_lens)
            # 传入blk[i]的注意力权重
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

# [have a try]
# 下⾯我们指定了超参数来创建⼀个2层8头注意力的Transformer编码器。Transformer编码器输出的形状
# 是（批量⼤⼩，时间步数⽬，num_hiddens）
# 回顾参数：TransformerEncoder(self, vocab_size, key_size, query_size, value_size,
#                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
#                 num_heads, num_layers, dropout, use_bias=False, **kwargs)
# encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
# encoder.eval()
# print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

#10.7.5 解码器
# Transformer解码器也是由多个相同的层组成。在DecoderBlock类中实现的每个层包含了三
# 个⼦层：解码器⾃注意⼒、“编码器-解码器”注意⼒和基于位置的前馈⽹络。这些⼦层也都被残差连接和
# 紧随的层规范化围绕。正如在本节前⾯所述，在掩蔽多头解码器⾃注意⼒层（第⼀个⼦层）中，查询、键和
# 值都来⾃上⼀个解码器层的输出。关于序列到序列模型（sequence-to-sequence model），在训练阶
# 段，其输出序列的所有位置（时间步）的词元都是已知的；然⽽，在预测阶段，其输出序列的词元是逐个⽣成的。
# 因此，在任何解码器时间步中，只有⽣成的词元才能⽤于解码器的⾃注意⼒计算中。为了在解码器中保留⾃回归
# 的属性，其掩蔽⾃注意⼒设定了参数dec_valid_lens，以便任何查询都只会与解码器中所有已经⽣成词元的
# 位置（即直到该查询位置为⽌）进⾏注意⼒计算。

# 在下面代码中，我们定义了一个 DecoderBlock 类，代表解码器中的第 i 个块。在 __init__ 方法中
# ，我们初始化了各个子模块，并将其保存为类属性。在 forward 方法中，我们首先获取输入的 X 和状态
# state，然后根据训练阶段和预测阶段的不同处理方式来选择性地组合 X 和 state[2][self.i]，生成
# key_values。接着，根据是否处于训练阶段，我们构造了适当的dec_valid_lens。然后，通过自注意力
# 机制，我们计算了 X 的新表示 X2，并与原始输入 X 进行残差连接和归一化得到 Y。接下来，使用编码
# 器-解码器注意力机制，我们计算了 Y 的新表示 Y2，并再次进行残差连接和归一化得到 Z。最后，我们
# 将 Z 输入到位置编码前馈神经网络（PositionWiseFFN）中进行变换，并将结果与 Z 进行残差连接和
# 归一化得到最终的输出，同时更新了 state。返回的第一个元素是输出结果，第二个元素是更新后的状态
# state。以上是代码的解释和注释。请注意，在代码中涉及到的 d2l.MultiHeadAttention、
# AddNorm 和 PositionWiseFFN 是自定义的模块。

#定义单个的Decoder模块
class DecoderBlock(nn.Module):   # 解码器块类，继承自nn.Module
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        #i相当于时间步,第i块解码器
        self.i = i
        # 创建多头注意力机制模块1，用于解码器自注意力
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        # 创建AddNorm层模块1，用于对解码器自注意力的输出进行残差连接和层归一化
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 创建多头注意力机制模块2，用于编码器-解码器注意力(注意与self.attention1不同)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        # 创建AddNorm层模块2，用于对编码器-解码器注意力的输出进行残差连接和层归一化
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 创建位置前馈神经网络模块，用于对编码器-解码器注意力的输出进行非线性变换
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        # 创建AddNorm层模块3，用于对位置前馈神经网络的输出进行残差连接和层归一化
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 三个参数self:实例对象。X：输入张量。state:状态。
        # state(enc_outputs, enc_valid_lens, dec_prev_outputs)
        # enc_outputs是编码器encoder的输出张量
        # enc_valid_lens是编码器encoder输入序列的有效长度
        # dec_prev_outputs是解码器decoder在当前时间步之前的输出表示（一个列表）
        # 1.在训练阶段，输出序列的所有词元都在同一时间处理，因此state[2][self.i]
        # 初始化为None。这里的state[2][self.i]是指解码器Decoder在当前时间步i之前的输出表示，
        # 而self.i表示当前Decoder块的索引。在训练阶段，state[2][self.i]被设置为None，
        # 表示当前Decoder块之前没有其他Decoder块的输出表示。
        # 2.在预测阶段，输出序列是通过词元一个接着一个解码的。state[2][self.i]
        # 包含着直到当前时间步第i个块解码的输出表示。所以在预测阶段，state[2][self.i]
        # 不为None，表示当前Decoder块之前已经有其他Decoder块的输出表示，并且将其与当前时间步
        # 的输入张量X进行拼接，得到key_values作为当前时间步的输入。
        # 训练阶段或者是解码器块的第一个时间步
        if state[2][self.i] is None:
            # 将输入X的值传给k值
            key_values = X
        # 预测阶段且不是解码器块的第一个时间步
        else:
            # 将之前的decoder的输出和 X 这两个张量在维度 1 上进行拼接
            key_values = torch.cat((state[2][self.i], X), axis=1)
            # 更新输出
            state[2][self.i] = key_values

        #判断是否在训练
        if self.training:
            # 训练阶段，有valid_lens，因为算第i个输出的时候，要把后面的mask
            # 第一个维度：表示批次大小(batch size)，即一次传入模型的样本数量。
            # 第二个维度：表示时间步数(time steps)，也就是序列的长度。每个时间步对应序列中的一个元素。
            # 第三个维度：表示特征维度(feature dimension)，即每个时间步上的特征数或特征向量的长度。
            batch_size, num_steps, _ = X.shape
            # 根据训练阶段与预测阶段的不同，设置解码器输入序列的有效长度dec_valid_lens：
            # 在训练阶段，将dec_valid_lens初始化为一个维度为[num_steps]的张量，
            # 同时，对这个生成的张量进行复制扩充，变为batch_size*[num_steps]的张量
            #从[1,...,num_steps]变为[1,...,num_steps，1,...,num_steps,...,1,...,num_steps]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 在预测阶段，将dec_valid_lens设为None。
            dec_valid_lens = None
        # 自注意力
        # X：解码器的输入序列。
        # state：解码器的状态
        # X2：通过自注意力机制得到的解码器的中间表示
        # Y：解码器经过自注意力机制和残差连接处理后的结果
        # Y2：解码器经过编码器 - 解码器注意力机制和残差连接处理后的结果
        # Z：解码器经过最后一层注意力和残差连接处理后的结果，
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        # attention1函数并不满足定义的参数传入。这是因为在d2l.MultiHeadAttention的实现中
        # forward方法接受的参数列表包括query, key, value和valid_lens，而不包括X。
        # 然而，self.attention1(X, key_values, key_values, dec_valid_lens)这行代
        # 码可以正常运行是因为在d2l.MultiHeadAttention的实现中，forward方法使用了*args
        # 语法来接受任意数量的位置参数。
        Y = self.addnorm1(X, X2)#self.addnorm1 = AddNorm(norm_shape, dropout)
        # 编码器-解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

# # [have a try]
# decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
# decoder_blk.eval()
# X = torch.ones((2, 100, 24))
# state = [encoder_blk(X, valid_lens), valid_lens, [None]]
# print(decoder_blk(X, state)[0].shape)

# 由num_layers个DecoderBlock实例组成的完整的Transformer解码器
# 定义Transformer解码器类，继承自AttentionDecoder
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        # 引用父类构造函数
        super(TransformerDecoder, self).__init__(**kwargs)
        #num_hiddens用于指定嵌入层和全连接层的维度大小（词向量的大小）
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        # 嵌入层，将输入的词索引转换为对应的嵌入向量
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        # 位置编码器，用于为词嵌入添加位置信息
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 创建num_layers个解码器块模块,编号分别是block[0],...,block[num_layers-1]
        # 加入已经编译好的Decoderblock模块
        self.blks = nn.Sequential()
        # 编号为[0,...,num_layers-1]
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        # 全连接层，将解码器块最后的隐藏状态映射到词表大小的输出
        self.dense = nn.Linear(num_hiddens, vocab_size)
    # 初始化解码器的状态
    # [None] * self.num_layers表示创建一个由self.num_layers个None组成的列表
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    # 对输入数据进行解码器块的前向计算，并记录每个解码器块的自注意力权重和"编码器-解码器"自注
    # 意力权重。然后，将解码器块的输出通过全连接层映射到输出词表的大小，并返回最终的输出和更新
    # 后的状态。
    def forward(self, X, state):
        # 将输入X中的词索引转换为对应的词嵌入向量,乘以math.sqrt(self.num_hiddens)，以缩放
        # 向量的值，用于防止在多头注意力计算中梯度爆炸的问题,调用self.pos_encoding()方法，
        # 给词嵌入向量添加位置编码。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 记录注意力权重。batch_size, num_steps, _ = X.shape,其中“_”是输入X的第三个维度
        # 第三个维度：表示特征维度(feature dimension)，即每个时间步上的特征数或特征向量的长度。
        # self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        self._attention_weights = []
        for _ in range(2):
            row = [None] * len(self.blks)
            self._attention_weights.append(row)
        for i, blk in enumerate(self.blks):
            # blk是一个成员变量，在__init__方法中被初始化为一个nn.Sequential()对象
            # （即一个包含多个子模块的顺序容器）。blk(X, state)表示对X和state输入
            # 进行解码器块的前向计算，得到输出和更新后的状态。i是一个索引值。
            X, state = blk(X, state)
            # self._attention_weights是一个二维列表，其中第一个列表记录了解码器块的自注意
            # 力权重，第二个列表记录了“编码器-解码器”自注意力权重。通过
            # self._attention_weights[0][i]，我们可以访问到第一个列表中的第i个位置。
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # "编码器-解码器"自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        # 使用全连接层将解码器块的输出映射到词表大小的输出
        return self.dense(X), state
    # 返回注意力权重
    @property
    def attention_weights(self):
        return self._attention_weights

# 10.7.6训练
# 指定Transformer的编码器和解码器都是2层，都使⽤4头注意⼒。为了进⾏序列到序列的学习，
# 下⾯在“英语－法语”机器翻译数据集上训练Transformer模型。
# 查询（Q）表示输入语句中的某个单词或子序列，用于计算注意力权重。在机器翻译任务中，
# （英语）中的单词或子序列。
# 键（K）表示源语言（英语）中的单词或子序列，用于与查询进行比较以计算注意力权重。
# 值（V）表示目标语言（法语）中的单词或子序列，通过与注意力权重相乘后加权求和，用于生成最终的翻译结果。
# num_steps:序列长度
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
#获取训练轮数，源语句，目标语句
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# # [have a try]
# # train_iter 是一个数据迭代器，用于在训练过程中按批次提供数据。它通常用于循环遍历训练数据集，
# # 将数据输入模型进行训练。要查看 train_iter 的内容，使用 iter() 函数将其转换为迭代器
# # 对象，并使用 next() 函数来一次获取一个批次的数据。将 train_iter 转换为迭代器对象
# train_iter_iter = iter(train_iter)
# # 显示第一个批次的数据
# batch = next(train_iter_iter)
# print(len(batch))
# # [have a try]
# # 查看源语句与目标语句中的词，按照index+src_token+tgt_token的形式输出
# for idx in range(len(src_vocab)):
#     token1 = src_vocab.idx_to_token[idx]
#     token2 = tgt_vocab.idx_to_token[idx]
#     print(f" Index: {idx},scr: {token1},tgt: {token2}")

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
#
# # 在代码中绘制损失与 epoch 的关系图
# # 假设 'loss_values' 是一个存储了每个 epoch 损失值的列表
# plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()  # 显示图形窗口

# # [have a try]
# # 训练结束后，使⽤Transformer模型将⼀些英语句⼦翻译成法语，并且计算它们的BLEU分数。
# engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# for eng, fra in zip(engs, fras):
#     translation, dec_attention_weight_seq = d2l.predict_seq2seq(
#             net, eng, src_vocab, tgt_vocab, num_steps, device, True)
# print(f'{eng} => {translation}, ',f'bleu {d2l.bleu(translation, fra, k=2):.3f}')



