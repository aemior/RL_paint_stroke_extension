# RL paint muilt stroke extention

这个项目是对 [LearningToPaint](https://github.com/megvii-research/ICCV2019-LearningToPaint) 的笔触拓展，增加了几种笔触并使用 [Stylized Neural Painting](https://github.com/jiupinjia/stylized-neural-painting) 的神经渲染器框架。项目中还包含笔触翻译器，可以实现不同种类笔触控制参数的转换。

## 安装

本项目在pytorch 1.9.0下测试通过，其他依赖参考 [LearningToPaint](https://github.com/megvii-research/ICCV2019-LearningToPaint) ，可能还需要visdom 0.1.8.9可视化训练过程

mypaint笔触参考[这里](./mypaint/libmypaint安装配置.md)的配置文件安装

## 训练

### Agent

训练Agent使用LearningToPaint 的代码

```
git clone https://github.com/megvii-research/ICCV2019-LearningToPaint.git
```

用[mod_DDPG](./mod_DDPG/ddpg.py)内的代码替换ICCV2019-LearningToPaint/baseline/DRL/ddpg.py，里面可以改笔触种类和渲染器的参数文件, 然后将 Networks/render.py 复制到 /ICCV2019-LearningToPaint/baseline/Renderer/ 文件夹内

为了保证代码正常运行，可能还需要修改ICCV2019-LearningToPaint/baseline/env.py 内的数据集路径，ICCV2019-LearningToPaint/baseline/utils/tensorboars.py内的图片转换方式

```python
from PIL import Image

img = Image.fromarray(img) #img = scipy.misc.toimage(img)
```

最后使用下面的代码来训练

```bash
cd baseline
python3 train.py --max_step=40 --debug --batch_size=96
```

