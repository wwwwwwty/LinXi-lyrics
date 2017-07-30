# LinXi-lyrics

> 使用RNN模型学习林夕的歌词数据，自动生成歌词。

## 项目结构
- data目录: 存放训练的歌词数据
- model目录: 保存训练模型。

- config.py: 训练超参数集合。
- preprocess.py: 歌词预处理，分词。
- get_batch.py: 生成训练所需的batch数据。
- rnn.py: 建立RNN模型。
- train.py: 训练，预测。
