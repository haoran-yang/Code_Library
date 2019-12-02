# data science code base
the code library for data science.

# 1. 文件名 visualization_utils.py
- EDA可视化通用方法汇总
## 1.1 连续特征
- 用于连续特征可视化
### 1.1.1 dist_numb_target
- 单个连续特征分布图(二分类)，y取值类型需为int型0,1
![dist_numb_target](https://img-blog.csdnimg.cn/20191106170122173.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
### 1.1.2 ploting_numb_fets
- 多连续特征分布（二分类）,参数draw_type:绘图类型,取值：dist_y(按y分布图)，dist(分布图)，box_y(按y增强箱形图)，box(增强箱形图)
![draw_type = "dist_y"](https://img-blog.csdnimg.cn/2019110617232482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
![draw_type = "dist"](https://img-blog.csdnimg.cn/20191106172559663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
![draw_type = "box_y"](https://img-blog.csdnimg.cn/20191106172646608.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
![draw_type = "box"](https://img-blog.csdnimg.cn/20191106173002485.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
### 1.1.3 dist_target_detail
- 连续特征基于target的区间分布。
- 可调节x轴区间范围。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106173344759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
## 1.2 类别特征
- 用于类别特征或离散特征可视化
### 1.2.1 dist_cate_target
- 单个类别特征分布绘图(二分类)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106170935856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
### 1.2.2 ploting_cat_fets
- 多类别特征或离散特征target正类分布，y取值类型需为int型0,1
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106171418651.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
## 1.3 目标分布
### 1.3.1 target_dist_plot
- 二分类目标分布
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106173251459.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
## 1.4 相关性
### 1.4.1 correlation_heatmap
- Pearson相关性热力图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106173853958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hhb3Jhbl95YW5n,size_16,color_FFFFFF,t_70)
