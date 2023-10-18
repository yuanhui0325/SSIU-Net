1.训练代码：train_manual1.py，测试代码：test_manual_y.py
2. 网络结构：model.py
3.prepareData文件夹中含制作数据集的代码（服务器上该代码位于weiwei/imageEnhancement/路径下），原序列来源于服务器/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/TMC/MPEG_CTC/Cat1_A路径下，编码后的序列位于服务器/media/vim/941bfed3-dbb1-4e73-80a7-a5601b4f9505/Wangweiwei/weiwei/TMC/compress_code/V9.0/cfg/octree-raht/lossless-geom-lossy-attrs路径下，制作好的数据集以h5文件的形式保存在./data/TrainData/路径下，根据不同码率分成了多个文件夹，在制作数据集时每个序列的数据均写为一个单独的h5文件，并以序列名命名。测试序列位于./data/OriData_test/yuv_format/sameorder下。
4.pths文件夹下存放了训练时保存的模型，每30个epoch保存了一次，以每天的日期为文件夹保存。
5.logs文件夹存放每次训练时保存的一些loss值等数据，以每天的日期为文件夹保存。
6.events文件夹存放tf产生的曲线图

网络可改进的地方：
U-Net网络结构的思想在于在提取每个点的特征时考虑其邻近点，但是用传统的二维卷积方法提取得到的特征，并不是MLP，具体可参考论文‘FPConv: Learning Local Flattening for Point Convolution’，代码参考https://github.com/lyqun/FPConv。该网络要求输入的是一整帧点云，可将一个序列用FPS算法多次提取1024个点。

