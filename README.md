
## 阿里妈妈国际广告算法大赛

https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409106.5678.1.704463d77Us11w&raceId=231647


### 主要特征

- 基本特征：给出的用户 店铺 商品 上下文等基本特征  包括预处理特征
- 交叉特征：各种基本特征的交叉占比
    - item 层面
    - brand 层面
    - city 层面
    - user 层面
    - shop 层面

- 统计特征：
    - 活跃度统计 :
        -user 活跃 item item_brand_id item_city_id   // 日活跃
        -user 活跃 shop ... // 日活跃
        -item 活跃 user
        -shop 活跃 user
    - 曝光度统计
        - 小时曝光度  user item
        - 时区曝光度  user item
    - 相关平均值统计
- trick特征: 点击时间差等
    - trick1：
        - 标记点击时间 首次 中间 最后一次
            - user 每天的点击时间
            - user 在 item 中每天的点击时间
            - user 在 brand 中每天点击时间
            - user 在 shop 中每天点击时间
    - 距离第一次点击的时间差：
        - user 每天点击时间差
        - item 被点击的时间差
        - user 点击 brand 时间差
        - user 点击 shop 时间差
    - 距离最后一次点击的时间差：
        - user 每天点击时间差
        - item 被点击的时间差
    - 距离 上/下 一次点击的时间差：
        - user item 两个

    - 10分钟内点击重复点击次数：
        - user
        - user 对 item
        - user 对 shop
    - user 不同时间段的点击次数 每天 每个小时 每隔10 、 15 、 30、 45分钟点击次数

- 特别特征：
    - 价格差
- 转化率特征：
    - 各个时间段的转化率统计  item city  brand shop  age user的转化率
    - 联合特征的转化率统计 （item age） （shop brand转化率）等等
    - 三特征联合转化率（shop age gener转化率统计）
### 特征选择工具 Wrapper法