# 大量图片高效处理

## 在照片中找出每个滑雪者

### 流程和细节优化
- Mask R-CNN
    - 跑完了万龙滑雪场 20190301 ~ 20190404 的照片
    - 共 86573 张照片 :muscle:
    - @MingxuanHu 爬来的数据 :clap:
- 多线程
    - 多线程读取图片
    - 多线程格式化数据
- 使用GPU
    - 图片处理速度提高 ~40x :rocket:
    - Titan X Pascal
        - `A 12GB GPU can typically handle 2 images of 1024x1024px`
    - 调整 `batch_size`
        - 2x GPU
        - 6x Image/GPU (700x467px, ~30%)
    - Tips [`tips_for_GPU.sh`](../utils/tips_for_GPU.sh)
- 高效结果存储
    - 减少空间使用 :floppy_disk:
        - **Masks**
            - Lists -> Strings
        - ROIs
            - Lists -> 4x Int
        - 减少 90% 内存消耗
            - `df.info(memory_usage="deep")`
    - Save DataFrame per batch
- 合并数据
    - 保存为**csv** :ledger:
        - 方便合并
        - 支持分 chunk 读取
    - 压缩
        - 压缩比很高，1000x
        - 因此压缩后的 csv 比 pkl 小多了
        - 可能因为Masks的数据类型0/1

### 数据处理

:pushpin:#TODO

<!--  -->
