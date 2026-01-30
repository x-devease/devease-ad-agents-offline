# 特征工程数据管道设计

## 📋 概述

### 设计目标
构建统一的特征工程管道，支持广告数据的批量处理和实时特征计算，为 Ad Miner（推荐引擎）和 Adset Allocator（预算分配）提供高质量特征。

### 数据类型
- **时间序列数据**: 广告表现随时间变化的指标
- **类别数据**: 广告属性、受众定位、创意类型
- **数值数据**: 预算、花费、转化等指标
- **图像数据**: 创意图片（通过 GPT-4 Vision 提取特征）

### 实施里程碑

```
Phase 1: Python 脚本 (MVP)
    ↓
Phase 2: Spark 批处理 (Production)
    ↓
Phase 3: 流处理 (Real-time)
```

---

## 🗺️ 实施里程碑

### Phase 1: Python 脚本 MVP

**目标**: 快速验证特征工程方法，建立基线

**技术栈**:
- Python 3.10+
- Pandas / NumPy
- Scikit-learn
- Jupyter Lab (开发)
- Python 脚本 (生产)

**交付物**:
```python
# scripts/feature_pipeline.py
python scripts/feature_pipeline.py \
  --customer customer_123 \
  --input data/raw/ad_data.csv \
  --output data/features/ \
  --features all
```

**特征列表**: 50-100 个核心特征

**处理能力**:
- 数据规模: < 10GB
- 处理时间: < 1 hour
- 调度: Cron 每日运行

**验收标准**:
- [ ] 能够处理 30 天历史数据
- [ ] 生成 50+ 特征
- [ ] 通过单元测试
- [ ] 文档完整

---

### Phase 2: Spark 批处理

**目标**: 生产级批处理，支持大规模数据

**技术栈**:
- Apache Spark 3.x (PySpark)
- AWS EMR / Dataproc
- Airflow 调度
- Parquet 存储
- MLflow 特征跟踪

**交付物**:
```python
# jobs/spark_feature_pipeline.py
spark-submit jobs/spark_feature_pipeline.py \
  --customer customer_123 \
  --input s3://data-bucket/raw/ \
  --output s3://features-bucket/ \
  --date 2025-01-29
```

**特征列表**: 200-300 个特征

**处理能力**:
- 数据规模: 100GB - 1TB
- 处理时间: < 30 min
- 调度: Airflow DAG

**验收标准**:
- [ ] 处理 1TB 数据 < 30 min
- [ ] 生成 200+ 特征
- [ ] 集成 Airflow
- [ ] 特征版本管理 (MLflow)

---

### Phase 3: 流处理

**目标**: 实时特征计算，支持在线决策

**技术栈**:
- AWS Lambda / Step Functions
- Amazon Kinesis
- Redis (在线特征存储)
- FastAPI (特征服务)
- Kafka (可选)

**交付物**:
```python
# Real-time Feature Service
curl -X POST https://features.api.example.com/update \
  -H "Content-Type: application/json" \
  -d '{"ad_id": "123", "event_type": "metrics_update", ...}'
```

**特征列表**: 300+ 特征（包括实时特征）

**处理能力**:
- 延迟: < 1 second
- 吞吐: 1000+ events/sec
- 可用性: 99.9%

**验收标准**:
- [ ] 端到端延迟 < 1 sec
- [ ] 支持 1000+ QPS
- [ ] 特征新鲜度 < 5 sec
- [ ] 完整监控和告警

---

## 🏗️ 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                    数据源层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Meta Ads API │  │ Historical   │  │ Webhook      │     │
│  │ (Real-time)  │  │ Exports      │  │ Events       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                    数据接入层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Batch        │  │ Stream       │  │ Image        │     │
│  │ Ingestion    │  │ Ingestion    │  │ Processing   │     │
│  │ Phase 1/2    │  │ Phase 3      │  │ (GPT-4V)     │     │
│  │ (Python/     │  │ (Kinesis)    │  │              │     │
│  │  Spark)      │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        │                                   │
        ↓                                   ↓
┌───────────────────────┐       ┌───────────────────────┐
│  批量特征管道          │       │  实时特征管道          │
│  Phase 1/2            │       │  Phase 3              │
│                       │       │                       │
│  - Python (MVP)       │       │  - Lambda/Step Func   │
│  - Spark (Prod)       │       │  - Incremental        │
└───────────────────────┘       └───────────────────────┘
        │                                   │
        └─────────────────┬─────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                  特征计算引擎                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  数值特征引擎     │  类别特征引擎   │  时序特征引擎  │  │
│  │  - 200+ 数值技术 │  - 100+ 类别技术│  - 150+ 时序技术│  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              交互特征引擎 (500+ 组合)                │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           特征名称混淆 (f1, f2, ...)                 │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│              特征名称映射层 (Privacy Layer)                 │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  f1: impressions_mean                              │  │
│  │  f2: ctr                                           │  │
│  │  f3: objective_format_combo                        │  │
│  │  ...                                               │  │
│  │  Total: 500+ features → f1...f500                  │  │
│  └─────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                  特征存储层 (Feature Store)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Online Store  │  │ Offline Store │  │ Metadata     │     │
│  │ (Redis/Dynamo)│  │ (S3/Parquet)  │  │ (MLflow)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│                    消费者层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Ad Miner     │  │ Adset        │  │ Analytics    │     │
│  │ (推荐引擎)    │  │ Allocator    │  │ Dashboard    │     │
│  │              │  │ (预算分配)    │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────────────────────────────┘
```

---

## 🔐 Part 1: 特征名称混淆（隐私保护）

### 1.1 设计原则

**目标**: 通过特征名称混淆保护客户数据隐私，同时保持内部可追溯性。

```python
# 映射示例
{
    # 内部名称 → 混淆名称
    "impressions_mean": "f1",
    "clicks_sum": "f2",
    "ctr": "f3",
    "roas_7d_avg": "f4",
    "objective_format_combo": "f5",
    "campaign_objective_encoding": "f6",
    ...
}

# 对客户/工程师暴露的特征向量
{
    "entity_id": "ad_123",
    "features": {
        "f1": 15000.5,
        "f2": 500,
        "f3": 3.33,
        "f4": 2.5,
        "f5": 0,
        "f6": 1,
        ...
    }
}
```

### 1.2 映射系统实现

```python
# src/pipelines/privacy/feature_name_obfuscator.py

import hashlib
import json
from typing import Dict, List, Optional
from pathlib import Path

class FeatureNameObfuscator:
    """
    特征名称混淆器

    原则:
    1. 确定性映射（同一名 → 同一 fX）
    2. 不可逆（fX 无法反推出原始名称）
    3. 可追溯（内部维护映射表）
    4. 版本控制（映射版本化）
    """

    def __init__(self, version: str = "1.0"):
        self.version = version
        self.mapping = self._load_mapping()

    def obfuscate(self, feature_name: str) -> str:
        """
        将特征名称混淆为 f1, f2, ...

        使用哈希确保确定性映射
        """
        # 计算哈希
        hash_value = int(hashlib.sha256(
            f"{feature_name}_{self.version}".encode()
        ).hexdigest(), 16)

        # 转换为 f1-f999
        feature_index = (hash_value % 999) + 1

        return f"f{feature_index}"

    def obfuscate_dict(self, features: Dict[str, any]) -> Dict[str, any]:
        """批量混淆特征字典"""
        obfuscated = {}
        mapping_record = {}

        for name, value in features.items():
            f_name = self.obfuscate(name)
            obfuscated[f_name] = value
            mapping_record[f_name] = name

        # 保存映射（仅内部访问）
        self._save_mapping_record(mapping_record)

        return obfuscated

    def deobfuscate(self, f_name: str) -> Optional[str]:
        """反混淆（仅内部使用）"""
        return self.mapping.get(f_name)

    def _load_mapping(self) -> Dict[str, str]:
        """加载映射表（从安全存储）"""
        mapping_file = Path(f"config/feature_mappings/v{self.version}.json")

        if mapping_file.exists():
            with open(mapping_file) as f:
                return json.load(f)
        else:
            return {}

    def _save_mapping_record(self, record: Dict[str, str]):
        """保存映射记录（到安全存储）"""
        mapping_file = Path("config/feature_mappings/internal.json")

        existing = {}
        if mapping_file.exists():
            with open(mapping_file) as f:
                existing = json.load(f)

        existing.update(record)

        with open(mapping_file, 'w') as f:
            json.dump(existing, f, indent=2)
```

### 1.3 映射表管理

```python
# config/feature_mappings/v1.0.json (仅内部可访问)
{
  "f1": "impressions_mean",
  "f2": "clicks_sum",
  "f3": "ctr",
  "f4": "roas_7d_avg",
  "f5": "objective_format_combo",
  "f6": "campaign_objective_encoding",
  "f7": "spend_rolling_std_7d",
  "f8": "impressions_lag_7d",
  ...
}

# 对外暴露的特征列表 (公开)
# config/feature_mappings/public_features.json
{
  "total_features": 500,
  "feature_list": ["f1", "f2", "f3", ..., "f500"],
  "feature_categories": {
    "numerical": ["f1", "f2", ..., "f250"],
    "categorical": ["f251", "f252", ..., "f350"],
    "timeseries": ["f351", "f352", ..., "f500"]
  }
}
```

### 1.4 使用示例

```python
# 使用混淆后的特征
obfuscator = FeatureNameObfuscator(version="1.0")

# 原始特征
raw_features = {
    "impressions_mean": 15000.5,
    "clicks_sum": 500,
    "ctr": 3.33,
    "roas_7d_avg": 2.5
}

# 混淆后（可以安全发送给客户）
obfuscated_features = obfuscator.obfuscate_dict(raw_features)
# {
#     "f1": 15000.5,
#     "f2": 500,
#     "f3": 3.33,
#     "f4": 2.5
# }

# 内部使用时可以反混淆
original_name = obfuscator.deobfuscate("f1")  # "impressions_mean"
```

---

## 🔢 Part 2: 全面的数值特征工程技术（200+ 特征）

### 2.1 基础统计特征（50+）

```python
class ComprehensiveNumericalFeatures:
    """全面的数值特征提取"""

    def extract_basic_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基础统计特征 (50+)

        包括:
        - 中心趋势: 均值、中位数、众数、几何均值、调和均值
        - 离散程度: 标准差、方差、范围、IQR、MAD
        - 分位数: p1, p5, p10, p25, p50, p75, p90, p95, p99
        - 分布形状: 偏度、峰度、Jarque-Bera 检验
        - 异常值: 异常值数量、比例、Z-score
        - 变换: Log, Box-Cox, Yeo-Johnson, Quantile
        """

        features = pd.DataFrame(index=df.index)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # 1. 中心趋势 (5)
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_median'] = df[col].median()
            features[f'{col}_mode'] = df[col].mode()[0] if not df[col].mode().empty else 0
            features[f'{col}_geometric_mean'] = self._geometric_mean(df[col])
            features[f'{col}_harmonic_mean'] = self._harmonic_mean(df[col])

            # 2. 离散程度 (8)
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_var'] = df[col].var()
            features[f'{col}_range'] = df[col].max() - df[col].min()
            features[f'{col}_iqr'] = df[col].quantile(0.75) - df[col].quantile(0.25)
            features[f'{col}_mad'] = df[col].mad()  # Mean Absolute Deviation
            features[f'{col}_cv'] = df[col].std() / (df[col].mean() + 1e-6)  # Coefficient of Variation
            features[f'{col}_range_coefficient'] = (df[col].max() - df[col].min()) / (df[col].mean() + 1e-6)
            features[f'{col}_quartile_coefficient'] = features[f'{col}_iqr'] / (df[col].quantile(0.75) + df[col].quantile(0.25) + 1e-6)

            # 3. 分位数 (9)
            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                features[f'{col}_p{q}'] = df[col].quantile(q/100)

            # 4. 分布形状 (5)
            features[f'{col}_skew'] = df[col].skew()
            features[f'{col}_kurtosis'] = df[col].kurtosis()
            features[f'{col}_jarque_bera'] = self._jarque_bera(df[col])
            features[f'{col}_excess_kurtosis'] = df[col].kurtosis()  # 超额峰度
            features[f'{col}_moment_5'] = ((df[col] - df[col].mean()) / df[col].std())**5  # 5阶矩

            # 5. 异常值 (4)
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
            features[f'{col}_outlier_count'] = outliers.sum()
            features[f'{col}_outlier_ratio'] = outliers.mean()
            features[f'{col}_outlier_mean_zscore'] = np.abs((df[col] - df[col].mean()) / df[col].std()).mean()
            features[f'{col}_extreme_outlier_ratio'] = ((np.abs((df[col] - df[col].mean()) / df[col].std()) > 3).mean())

            # 6. 数据变换 (6)
            features[f'{col}_log'] = np.log1p(df[col])
            features[f'{col}_log2'] = np.log2(df[col] + 1)
            features[f'{col}_log10'] = np.log10(df[col] + 1)
            features[f'{col}_sqrt'] = np.sqrt(df[col].abs())
            features[f'{col}_boxcox'], _ = self._boxcox_transform(df[col])
            features[f'{col}_yeojohnson'], _ = self._yeojohnson_transform(df[col])

            # 7. 归一化 (4)
            min_val, max_val = df[col].min(), df[col].max()
            features[f'{col}_minmax'] = (df[col] - min_val) / (max_val - min_val + 1e-6)
            features[f'{col}_robust'] = (df[col] - df[col].median()) / (features[f'{col}_iqr'] + 1e-6)
            features[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            features[f'{col}_unit_vector'] = df[col] / (np.linalg.norm(df[col]) + 1e-6)

            # 8. 百分位排名 (2)
            features[f'{col}_percentile_rank'] = df[col].rank(pct=True)
            features[f'{col}_decile_rank'] = pd.cut(df[col].rank(pct=True), bins=10, labels=False)

        return features

    def _geometric_mean(self, series: pd.Series) -> float:
        """几何均值"""
        return np.exp(np.log(series[series > 0]).mean()) if (series > 0).any() else 0

    def _harmonic_mean(self, series: pd.Series) -> float:
        """调和均值"""
        return len(series) / np.sum(1.0 / (series + 1e-6))

    def _jarque_bera(self, series: pd.Series) -> float:
        """Jarque-Bera 正态性检验"""
        from scipy.stats import jarque_bera
        return jarque_bera(series.dropna())[0]

    def _boxcox_transform(self, series: pd.Series):
        """Box-Cox 变换"""
        from scipy.stats import boxcox
        try:
            transformed, _ = boxcox(series + 1 - series.min())
            return transformed.mean(), _
        except:
            return series.mean(), 0

    def _yeojohnson_transform(self, series: pd.Series):
        """Yeo-Johnson 变换"""
        from sklearn.preprocessing import PowerTransformer
        try:
            pt = PowerTransformer(method='yeo-johnson')
            transformed = pt.fit_transform(series.values.reshape(-1, 1))
            return transformed.mean(), pt.lambdas_[0]
        except:
            return series.mean(), 0
```

### 2.2 高级统计特征（40+）

```python
    def extract_advanced_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        高级统计特征 (40+)

        包括:
        - 熵和互信息
        - 相关系数
        - 累积统计
        - 百分位变化
        - 相对差异
        - 字符串相似度 (用于类别编码后的数值)
        """

        features = pd.DataFrame(index=df.index)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # 1. 熵特征 (3)
            features[f'{col}_entropy'] = self._calculate_entropy(df[col])
            features[f'{col}_conditional_entropy'] = self._conditional_entropy(df[col], df.get('roas', pd.Series()))
            features[f'{col}_mutual_info'] = self._mutual_info(df[col], df.get('conversions', pd.Series()))

            # 2. 相关性 (3)
            if 'roas' in df.columns:
                features[f'{col}_correlation_with_roas'] = df[col].corr(df['roas'])
            if 'spend' in df.columns:
                features[f'{col}_correlation_with_spend'] = df[col].corr(df['spend'])
            if 'impressions' in df.columns:
                features[f'{col}_correlation_with_impressions'] = df[col].corr(df['impressions'])

            # 3. 累积统计 (5)
            features[f'{col}_cumsum'] = df[col].cumsum()
            features[f'{col}_cummax'] = df[col].cummax()
            features[f'{col}_cummin'] = df[col].cummin()
            features[f'{col}_cummean'] = df[col].expanding().mean()
            features[f'{col}_cumstd'] = df[col].expanding().std()

            # 4. 百分位变化 (4)
            features[f'{col}_pct_change_1'] = df[col].pct_change(1)
            features[f'{col}_pct_change_7'] = df[col].pct_change(7)
            features[f'{col}_pct_change_30'] = df[col].pct_change(30)
            features[f'{col}_pct_change_90'] = df[col].pct_change(90)

            # 5. 相对差异 (4)
            features[f'{col}_diff_from_mean'] = df[col] - df[col].mean()
            features[f'{col}_diff_from_median'] = df[col] - df[col].median()
            features[f'{col}_pct_diff_from_mean'] = ((df[col] - df[col].mean()) / (df[col].mean() + 1e-6)) * 100
            features[f'{col}_pct_diff_from_median'] = ((df[col] - df[col].median()) / (df[col].median() + 1e-6)) * 100

            # 6. 加权统计 (3)
            weights = df.get('impressions', pd.Series([1]*len(df)))
            features[f'{col}_weighted_mean'] = np.average(df[col], weights=weights)
            features[f'{col}_weighted_std'] = np.sqrt(np.average((df[col] - features[f'{col}_weighted_mean'])**2, weights=weights))
            features[f'{col}_weighted_sum'] = (df[col] * weights).sum()

            # 7. 缩放统计 (3)
            features[f'{col}_sum_squares'] = (df[col] ** 2).sum()
            features[f'{col}_norm_l1'] = np.abs(df[col]).sum()
            features[f'{col}_norm_l2'] = np.sqrt((df[col] ** 2).sum())

        return features

    def _calculate_entropy(self, series: pd.Series, n_bins: int = 10) -> float:
        """计算熵"""
        counts, _ = np.histogram(series.dropna(), bins=n_bins)
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _conditional_entropy(self, x: pd.Series, y: pd.Series) -> float:
        """条件熵"""
        if y.empty:
            return 0
        # 简化实现
        return self._calculate_entropy(x) * 0.8

    def _mutual_info(self, x: pd.Series, y: pd.Series) -> float:
        """互信息"""
        from sklearn.metrics import mutual_info_score
        if y.empty:
            return 0
        # 离散化
        x_discrete = pd.cut(x, bins=10, labels=False)
        y_discrete = pd.cut(y, bins=10, labels=False)
        return mutual_info_score(x_discrete, y_discrete)
```

### 2.3 时间序列数值特征（60+）

```python
    def extract_timeseries_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间序列数值特征 (60+)

        包括:
        - 滚动窗口统计 (各种窗口大小)
        - 滞后特征
        - 差分特征
        - 百分比变化
        - 加速度
        - 动量
        - 趋势
        - 波动率
        - 自相关
        """

        features = pd.DataFrame(index=df.index)
        numeric_cols = ['impressions', 'clicks', 'spend', 'conversions', 'roas']

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # 1. 滚动窗口 - 中心趋势 (12)
            for window in [3, 7, 14, 30]:
                features[f'{col}_rolling_mean_{window}d'] = df[col].rolling(window).mean()
                features[f'{col}_rolling_median_{window}d'] = df[col].rolling(window).median()
                features[f'{col}_rolling_expmean_{window}d'] = df[col].ewm(span=window).mean()

            # 2. 滚动窗口 - 离散程度 (12)
            for window in [3, 7, 14, 30]:
                features[f'{col}_rolling_std_{window}d'] = df[col].rolling(window).std()
                features[f'{col}_rolling_var_{window}d'] = df[col].rolling(window).var()
                features[f'{col}_rolling_range_{window}d'] = df[col].rolling(window).max() - df[col].rolling(window).min()

            # 3. 滚动窗口 - 累积 (8)
            for window in [7, 14, 30]:
                features[f'{col}_rolling_sum_{window}d'] = df[col].rolling(window).sum()
                features[f'{col}_rolling_min_{window}d'] = df[col].rolling(window).min()
                features[f'{col}_rolling_max_{window}d'] = df[col].rolling(window).max()

            # 4. 滞后特征 (10)
            for lag in [1, 2, 3, 7, 14, 30, 60, 90]:
                features[f'{col}_lag_{lag}d'] = df[col].shift(lag)

            # 5. 差分特征 (5)
            features[f'{col}_diff_1'] = df[col].diff(1)
            features[f'{col}_diff_7'] = df[col].diff(7)
            features[f'{col}_diff_30'] = df[col].diff(30)
            features[f'{col}_diff_pct_1'] = df[col].pct_change(1)
            features[f'{col}_diff_pct_7'] = df[col].pct_change(7)

            # 6. 加速度和动量 (4)
            features[f'{col}_acceleration'] = df[col].diff(1).diff(1)
            features[f'{col}_momentum_7d'] = df[col] - df[col].shift(7)
            features[f'{col}_momentum_30d'] = df[col] - df[col].shift(30)
            features[f'{col}_roc_7d'] = ((df[col] - df[col].shift(7)) / df[col].shift(7)) * 100  # Rate of Change

            # 7. 波动率 (4)
            for window in [7, 14, 30]:
                features[f'{col}_volatility_{window}d'] = df[col].pct_change().rolling(window).std()
                features[f'{col}_volatility_exp_{window}d'] = df[col].pct_change().ewm(span=window).std()

            # 8. 自相关 (3)
            for lag in [1, 7, 14]:
                features[f'{col}_autocorr_lag{lag}'] = df[col].autocorr(lag=lag)

            # 9. 趋势特征 (3)
            features[f'{col}_trend_slope_7d'] = df[col].rolling(7).apply(self._linear_slope)
            features[f'{col}_trend_r2_7d'] = df[col].rolling(7).apply(self._trend_r2)
            features[f'{col}_trend_strength'] = df[col].rolling(30).apply(lambda x: np.corrcoef(range(len(x)), x)[0, 1]**2 if len(x) > 1 else 0)

        return features

    def _linear_slope(self, series):
        """计算线性趋势斜率"""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]

    def _trend_r2(self, series):
        """计算趋势的 R²"""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((series - y_pred) ** 2)
        ss_tot = np.sum((series - series.mean()) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-6))
```

### 2.4 比率和衍生特征（30+）

```python
    def extract_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        比率和衍生特征 (30+)

        包括:
        - 效率指标 (CTR, CVR, ROAS, CPA, CPC, CPM)
        - 复合比率
        - 相对比率
        - 百分位比率
        - 倒数特征
        """

        features = pd.DataFrame(index=df.index)

        # 1. 基础效率指标 (7)
        features['ctr'] = (df['clicks'] / (df['impressions'] + 1e-6)) * 100
        features['cvr'] = (df['conversions'] / (df['clicks'] + 1e-6)) * 100
        features['roas'] = df['revenue'] / (df['spend'] + 1e-6)
        features['cpa'] = df['spend'] / (df['conversions'] + 1e-6)
        features['cpc'] = df['spend'] / (df['clicks'] + 1e-6)
        features['cpm'] = (df['spend'] / (df['impressions'] + 1e-6)) * 1000
        features['rpm'] = (df['revenue'] / (df['impressions'] + 1e-6)) * 1000  # Revenue per Mille

        # 2. 复合比率 (8)
        features['roas_per_click'] = df['revenue'] / (df['clicks'] + 1e-6)
        features['revenue_per_impression'] = df['revenue'] / (df['impressions'] + 1e-6)
        features['cost_per_impression'] = df['spend'] / (df['impressions'] + 1e-6)
        features['conversion_value'] = df['revenue'] / (df['conversions'] + 1e-6)
        features['click_to_conversion_ratio'] = df['conversions'] / (df['clicks'] + 1e-6)
        features['spend_to_budget_ratio'] = df['spend'] / (df['budget'] + 1e-6)
        features['impressions_per_click'] = df['impressions'] / (df['clicks'] + 1e-6)
        features['cost_efficiency'] = df['revenue'] / (df['spend'] + 1e-6)  # Same as ROAS

        # 3. 相对比率 (5)
        features['ctr_vs_benchmark'] = features['ctr'] / (features['ctr'].mean() + 1e-6)
        features['cvr_vs_benchmark'] = features['cvr'] / (features['cvr'].mean() + 1e-6)
        features['roas_vs_benchmark'] = features['roas'] / (features['roas'].mean() + 1e-6)
        features['cpm_vs_benchmark'] = features['cpm'] / (features['cpm'].mean() + 1e-6)
        features['efficiency_score'] = (features['ctr'] * features['cvr'] * features['roas']) ** (1/3)

        # 4. 百分位比率 (4)
        for col in ['ctr', 'cvr', 'roas', 'cpm']:
            percentile = df[col].rank(pct=True)
            features[f'{col}_percentile'] = percentile
            features[f'{col}_is_top_10pct'] = (percentile >= 0.9).astype(int)
            features[f'{col}_is_top_25pct'] = (percentile >= 0.75).astype(int)
            features[f'{col}_is_bottom_25pct'] = (percentile <= 0.25).astype(int)

        # 5. 倒数特征 (3)
        features['impressions_inverse'] = 1 / (df['impressions'] + 1e-6)
        features['spend_inverse'] = 1 / (df['spend'] + 1e-6)
        features['cost_per_unit_inverse'] = 1 / (features['cpa'] + 1e-6)

        return features
```

### 2.5 稳定性和异常检测特征（20+）

```python
    def extract_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        稳定性和异常检测特征 (20+)

        包括:
        - 波动率指标
        - 稳定性得分
        - 变化点检测
        - 异常分数
        """

        features = pd.DataFrame(index=df.index)
        numeric_cols = ['impressions', 'clicks', 'spend', 'roas']

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # 1. 波动率指标 (8)
            features[f'{col}_volatility_7d'] = df[col].rolling(7).std() / (df[col].rolling(7).mean() + 1e-6)
            features[f'{col}_volatility_14d'] = df[col].rolling(14).std() / (df[col].rolling(14).mean() + 1e-6)
            features[f'{col}_volatility_30d'] = df[col].rolling(30).std() / (df[col].rolling(30).mean() + 1e-6)
            features[f'{col}_max_drawdown'] = df[col].rolling(30).apply(lambda x: (x.max() - x.min()) / x.max())
            features[f'{col}_price_variation'] = (df[col].rolling(30).max() - df[col].rolling(30).min()) / (df[col].rolling(30).mean() + 1e-6)
            features[f'{col}_avg_directional_change'] = np.abs(df[col].diff(1)).mean()
            features[f'{col}_volatility_ratio_7_30'] = features[f'{col}_volatility_7d'] / (features[f'{col}_volatility_30d'] + 1e-6)
            features[f'{col}_stability_index'] = 1 / (1 + features[f'{col}_volatility_30d'])

            # 2. 稳定性特征 (4)
            features[f'{col}_consecutive_up'] = (df[col].diff(1) > 0).astype(int).groupby((df[col].diff(1) <= 0).cumsum()).cumsum()
            features[f'{col}_consecutive_down'] = (df[col].diff(1) < 0).astype(int).groupby((df[col].diff(1) >= 0).cumsum()).cumsum()
            features[f'{col}_direction_changes'] = ((df[col].diff(1) > 0) != (df[col].diff(1).shift(1) > 0)).astype(int).cumsum()
            features[f'{col}_stability_score'] = 1 - (features[f'{col}_direction_changes'] / len(df))

            # 3. 异常检测 (5)
            rolling_mean = df[col].rolling(30).mean()
            rolling_std = df[col].rolling(30).std()
            features[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-6)
            features[f'{col}_is_anomaly_zscore'] = (np.abs(features[f'{col}_zscore']) > 3).astype(int)

            rolling_Q1 = df[col].rolling(30).quantile(0.25)
            rolling_Q3 = df[col].rolling(30).quantile(0.75)
            rolling_IQR = rolling_Q3 - rolling_Q1
            features[f'{col}_iqr_anomaly'] = ((df[col] < (rolling_Q1 - 1.5 * rolling_IQR)) | (df[col] > (rolling_Q3 + 1.5 * rolling_IQR))).astype(int)

            features[f'{col}_anomaly_score'] = np.abs(features[f'{col}_zscore']) * features[f'{col}_iqr_anomaly']

        return features
```

---

## 🏷️ Part 3: 全面的类别特征工程技术（100+ 特征）

### 3.1 基础编码技术（20+）

```python
class ComprehensiveCategoricalFeatures:
    """全面的类别特征提取"""

    def extract_basic_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基础编码技术 (20+)

        包括:
        - Label Encoding
        - One-Hot Encoding
        - Binary Encoding
        - BaseN Encoding
        - Ordinal Encoding
        """

        features = pd.DataFrame(index=df.index)
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            # 1. Label Encoding (1)
            le = LabelEncoder()
            features[f'{col}_label'] = le.fit_transform(df[col].fillna('Unknown'))

            # 2. One-Hot Encoding (针对低基数，< 10)
            if df[col].nunique() < 10:
                dummies = pd.get_dummies(df[col].fillna('Unknown'), prefix=f'{col}_onehot')
                features = pd.concat([features, dummies], axis=1)

            # 3. Binary Encoding (针对中基数，10-100)
            if 10 <= df[col].nunique() < 100:
                be = ce.BinaryEncoder(cols=[col])
                binary_encoded = be.fit_transform(df[[col]].fillna('Unknown'))
                features = pd.concat([features, binary_encoded], axis=1)

            # 4. BaseN Encoding (Base5)
            if df[col].nunique() < 50:
                bne = ce.BaseNEncoder(base=5, cols=[col])
                basen_encoded = bne.fit_transform(df[[col]].fillna('Unknown'))
                features = pd.concat([features, basen_encoded], axis=1)

            # 5. Ordinal Encoding (针对有序类别)
            if self._is_ordinal(col):
                oe = ce.OrdinalEncoder(cols=[col])
                features[f'{col}_ordinal'] = oe.fit_transform(df[[col]].fillna('Unknown'))

        return features

    def _is_ordinal(self, col: str) -> bool:
        """判断是否为有序类别"""
        ordinal_cols = ['targeting_age_range', 'video_length', 'ad_format']
        return any(ord_col in col for ord_col in ordinal_cols)
```

### 3.2 目标编码技术（15+）

```python
    def extract_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        目标编码技术 (15+)

        包括:
        - Target Encoding (Mean Encoding)
        - Smoothed Target Encoding
        - Leave-One-Out Target Encoding
        - M-Estimate Encoding
        - WOE (Weight of Evidence) Encoding
        """

        features = pd.DataFrame(index=df.index)
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            # 1. Target Encoding (1)
            target_mean = df.groupby(col)['roas'].mean()
            features[f'{col}_target_enc'] = df[col].map(target_mean).fillna(df['roas'].mean())

            # 2. Smoothed Target Encoding (1)
            smoothing_factor = 10
            count = df[col].value_counts()
            global_mean = df['roas'].mean()
            smoothed_mean = (count * features[f'{col}_target_enc'] + smoothing_factor * global_mean) / (count + smoothing_factor)
            features[f'{col}_target_enc_smooth'] = df[col].map(smoothed_mean)

            # 3. Leave-One-Out Target Encoding (1)
            loo_mean = df.groupby(col)['roas'].transform(lambda x: (x.sum() - x) / (len(x) - 1))
            features[f'{col}_target_enc_loo'] = loo_mean.fillna(df['roas'].mean())

            # 4. M-Estimate Encoding (1)
            m = 100
            m_estimate = (count * features[f'{col}_target_enc'] + m * global_mean) / (count + m)
            features[f'{col}_m_estimate'] = df[col].map(m_estimate)

            # 5. WOE Encoding (针对二分类问题) (1)
            # 这里简化为基于 ROAS > threshold 的二分类
            df['high_roas'] = (df['roas'] > df['roas'].median()).astype(int)
            woe = self._calculate_woe(df[col], df['high_roas'])
            features[f'{col}_woe'] = df[col].map(woe).fillna(0)

            # 6. Target Encoding with CV (1)
            # K-Fold 目标编码防止过拟合
            features[f'{col}_target_enc_cv'] = self._kfold_target_encoding(df, col, 'roas', k=5)

        return features

    def _calculate_woe(self, categorical: pd.Series, target: pd.Series) -> dict:
        """计算 Weight of Evidence"""
        woe_dict = {}
        for category in categorical.unique():
            cat_data = target[categorical == category]
            pos = cat_data.sum()
            neg = len(cat_data) - pos
            total_pos = target.sum()
            total_neg = len(target) - total_pos

            if pos == 0 or neg == 0:
                woe_dict[category] = 0
            else:
                woe_dict[category] = np.log((pos / total_pos) / (neg / total_neg))
        return woe_dict

    def _kfold_target_encoding(self, df: pd.DataFrame, cat_col: str, target_col: str, k: int = 5) -> pd.Series:
        """K-Fold 目标编码"""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        encoded = pd.Series(index=df.index, dtype=float)

        for train_idx, val_idx in kf.split(df):
            train_mean = df.iloc[train_idx].groupby(cat_col)[target_col].mean()
            encoded.iloc[val_idx] = df.iloc[val_idx][cat_col].map(train_mean)

        return encoded.fillna(df[target_col].mean())
```

### 3.3 频率和计数编码（10+）

```python
    def extract_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        频率和计数编码 (10+)

        包括:
        - Count Encoding
        - Frequency Encoding
        - Target Count Encoding
        - Cumulative Count Encoding
        - Rare Category Encoding
        """

        features = pd.DataFrame(index=df.index)
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            # 1. Count Encoding (1)
            count = df[col].value_counts()
            features[f'{col}_count'] = df[col].map(count)

            # 2. Frequency Encoding (1)
            features[f'{col}_freq'] = df[col].map(count / len(df))

            # 3. Log Frequency (1)
            features[f'{col}_log_freq'] = np.log1p(features[f'{col}_count'])

            # 4. Target Count (1)
            target_count = df.groupby(col)['roas'].count()
            features[f'{col}_target_count'] = df[col].map(target_count)

            # 5. Cumulative Count (1)
            features[f'{col}_cumcount'] = df.groupby(col).cumcount() + 1

            # 6. Rare Category Indicator (1)
            rare_threshold = 0.01  # 出现频率 < 1%
            features[f'{col}_is_rare'] = (features[f'{col}_freq'] < rare_threshold).astype(int)

            # 7. Category Frequency Rank (1)
            features[f'{col}_freq_rank'] = df[col].map(count.rank(ascending=False))

            # 8. Category Density (1)
            features[f'{col}_density'] = df[col].map(count / len(df))

        return features
```

### 3.4 嵌入和相似度编码（15+）

```python
    def extract_embedding_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        嵌入和相似度编码 (15+)

        包括:
        - Entity Embedding
        - TF-IDF (针对文本)
        - Similarity Encoding
        - Hashing Encoding
        - Polynomial Coding
        """

        features = pd.DataFrame(index=df.index)

        # 1. Hashing Encoder (针对高基数) (3)
        high_card_cols = ['ad_id', 'adset_id', 'campaign_id']
        for col in high_card_cols:
            if col in df.columns:
                he = ce.HashingEncoder(cols=[col], n_components=8)
                hashed = he.fit_transform(df[[col]].fillna('Unknown'))
                features = pd.concat([features, hashed], axis=1)

        # 2. Text TF-IDF Features (针对广告描述) (5)
        if 'ad_description' in df.columns:
            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
            tfidf_matrix = tfidf.fit_transform(df['ad_description'].fillna(''))

            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'desc_tfidf_{i}' for i in range(20)],
                index=df.index
            )
            features = pd.concat([features, tfidf_df], axis=1)

        # 3. Polynomial Coding (针对有序类别) (2)
        ordinal_cols = [col for col in df.columns if self._is_ordinal(col)]
        for col in ordinal_cols:
            n_categories = df[col].nunique()
            for i in range(min(3, n_categories - 1)):
                features[f'{col}_poly_{i}'] = (pd.factorize(df[col].fillna('Unknown'))[0] ** i)

        return features
```

---

## 🔄 Part 4: 全面的交互特征工程（500+ 组合）

### 4.1 数值-数值交互（200+）

```python
class ComprehensiveInteractionFeatures:
    """全面的交互特征提取"""

    def extract_numerical_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数值-数值交互特征 (200+)

        包括:
        - 算术交互: +, -, *, /
        - 多项式交互
        - 比率交互
        - 对数交互
        - 指数交互
        - 分段交互
        """

        features = pd.DataFrame(index=df.index)
        numeric_cols = ['impressions', 'clicks', 'spend', 'conversions', 'revenue', 'roas', 'ctr', 'cvr']

        # 1. 算术交互: 加法 (21) - nC2
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]

        # 2. 算术交互: 乘法 (21)
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                features[f'{col1}_times_{col2}'] = df[col1] * df[col2]

        # 3. 算术交互: 减法 (21)
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                features[f'{col2}_minus_{col1}'] = df[col2] - df[col1]

        # 4. 算术交互: 除法 (42)
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols:
                if col1 != col2:
                    features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-6)

        # 5. 多项式交互: 二次 (7)
        for col in numeric_cols:
            features[f'{col}_squared'] = df[col] ** 2

        # 6. 多项式交互: 三次 (7)
        for col in numeric_cols:
            features[f'{col}_cubed'] = df[col] ** 3

        # 7. 多项式交互: 平方根 (7)
        for col in numeric_cols:
            features[f'{col}_sqrt'] = np.sqrt(df[col].abs())

        # 8. 对数交互 (14)
        for col1, col2 in [('impressions', 'clicks'), ('spend', 'conversions'), ('revenue', 'spend')]:
            features[f'{col1}_log_plus_{col2}_log'] = np.log1p(df[col1]) + np.log1p(df[col2])
            features[f'{col1}_log_times_{col2}_log'] = np.log1p(df[col1]) * np.log1p(df[col2])

        # 9. 指数交互 (7)
        for col in numeric_cols[:3]:  # 只对前3个特征计算
            features[f'{col}_exp'] = np.exp(df[col] / (df[col].max() + 1e-6))  # 归一化防止溢出

        # 10. 分段交互 (14)
        for col1, col2 in [('impressions', 'spend'), ('clicks', 'conversions')]:
            # 高-高组合
            features[f'{col1}_high_{col2}_high'] = ((df[col1] > df[col1].median()) & (df[col2] > df[col2].median())).astype(int)
            # 高-低组合
            features[f'{col1}_high_{col2}_low'] = ((df[col1] > df[col1].median()) & (df[col2] <= df[col2].median())).astype(int)
            # 低-高组合
            features[f'{col1}_low_{col2}_high'] = ((df[col1] <= df[col1].median()) & (df[col2] > df[col2].median())).astype(int)
            # 低-低组合
            features[f'{col1}_low_{col2}_low'] = ((df[col1] <= df[col1].median()) & (df[col2] <= df[col2].median())).astype(int)

        return features
```

### 4.2 类别-类别交互（150+）

```python
    def extract_categorical_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        类别-类别交互特征 (150+)

        包括:
        - 组合特征
        - 交叉统计
        - 条件概率
        - 共现模式
        """

        features = pd.DataFrame(index=df.index)
        categorical_cols = ['campaign_objective', 'ad_format', 'targeting_gender', 'targeting_age_range', 'call_to_action']

        # 1. 两两组合 (10) - nC2 for 5 cols
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                features[f'{col1}_{col2}_combo'] = df[col1].astype(str) + '_' + df[col2].astype(str)

        # 2. 三三组合 (10) - nC3
        import itertools
        for col1, col2, col3 in itertools.combinations(categorical_cols, 3):
            features[f'{col1}_{col2}_{col3}_combo'] = df[col1].astype(str) + '_' + df[col2].astype(str) + '_' + df[col3].astype(str)

        # 3. 交互统计 (针对组合特征) (50)
        combo_features = [col for col in features.columns if '_combo' in col]
        for combo in combo_features[:5]:  # 只对前5个组合计算
            # 组合的频率
            combo_freq = features[combo].value_counts(normalize=True)
            features[f'{combo}_freq'] = features[combo].map(combo_freq)

            # 组合的目标均值
            if 'roas' in df.columns:
                combo_target_mean = df.groupby(features[combo])['roas'].mean()
                features[f'{combo}_target_mean'] = features[combo].map(combo_target_mean)

        # 4. 条件概率 (30)
        for col1, col2 in itertools.combinations(categorical_cols[:4], 2):  # 4个类别取2个
            # P(col2 | col1)
            conditional_prob = df.groupby(col1)[col2].value_counts(normalize=True)
            features[f'{col2}_given_{col1}_prob'] = df.apply(
                lambda row: conditional_prob.get((row[col1], row[col2]), 0),
                axis=1
            )

        # 5. 共现指标 (20)
        # 计算两个类别特征同时出现的强度
        for col1, col2 in itertools.combinations(categorical_cols[:5], 2):
            # Pointwise Mutual Information
            p_col1 = df[col1].value_counts(normalize=True)
            p_col2 = df[col2].value_counts(normalize=True)
            p_col1_col2 = df.groupby([col1, col2]).size() / len(df)

            pmi = []
            for _, row in df.iterrows():
                p_xy = p_col1_col2.get((row[col1], row[col2]), 1e-10)
                p_x = p_col1.get(row[col1], 1e-10)
                p_y = p_col2.get(row[col2], 1e-10)
                pmi_val = np.log2(p_xy / (p_x * p_y + 1e-10))
                pmi.append(pmi_val)

            features[f'{col1}_{col2}_pmi'] = pmi

        return features
```

### 4.3 数值-类别交互（100+）

```python
    def extract_numerical_categorical_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数值-类别交互特征 (100+)

        包括:
        - 按类别分组的数值统计
        - 类别条件下的数值特征
        - 数值-类别组合编码
        """

        features = pd.DataFrame(index=df.index)
        numeric_cols = ['impressions', 'clicks', 'spend', 'roas']
        categorical_cols = ['campaign_objective', 'ad_format', 'targeting_gender']

        # 1. 按类别分组的数值统计 (36) - 3 cat * 4 num * 3 stats
        for cat_col in categorical_cols:
            if cat_col not in df.columns:
                continue
            for num_col in numeric_cols:
                if num_col not in df.columns:
                    continue

                # 每个类别的均值
                group_mean = df.groupby(cat_col)[num_col].transform('mean')
                features[f'{num_col}_mean_by_{cat_col}'] = group_mean

                # 每个类别的标准差
                group_std = df.groupby(cat_col)[num_col].transform('std')
                features[f'{num_col}_std_by_{cat_col}'] = group_std

                # 每个类别的排名
                group_rank = df.groupby(cat_col)[num_col].rank(pct=True)
                features[f'{num_col}_rank_in_{cat_col}'] = group_rank

        # 2. 数值与类别的偏差 (24)
        for cat_col in categorical_cols:
            if cat_col not in df.columns:
                continue
            for num_col in numeric_cols:
                if num_col not in df.columns:
                    continue

                group_mean = df.groupby(cat_col)[num_col].transform('mean')
                features[f'{num_col}_deviation_from_{cat_col}_mean'] = df[num_col] - group_mean
                features[f'{num_col}_ratio_to_{cat_col}_mean'] = df[num_col] / (group_mean + 1e-6)

        # 3. 类别条件下的数值特征 (30)
        for cat_col in categorical_cols:
            if cat_col not in df.columns:
                continue
            for num_col in numeric_cols:
                if num_col not in df.columns:
                    continue

                # 类别 one-hot 后与数值的乘积
                for category in df[cat_col].unique()[:3]:  # 限制类别数
                    category_encoded = (df[cat_col] == category).astype(int)
                    features[f'{num_col}_for_{cat_col}_{category}'] = df[num_col] * category_encoded

        # 4. 数值-类别组合目标编码 (12)
        if 'roas' in df.columns:
            for cat_col in categorical_cols[:2]:  # 只对前2个类别
                if cat_col not in df.columns:
                    continue
                for num_col in numeric_cols[:2]:  # 只对前2个数值
                    if num_col not in df.columns:
                        continue

                    # 数值分桶 + 类别组合的目标编码
                    df[f'{num_col}_binned'] = pd.cut(df[num_col], bins=5, labels=False)
                    combo_target_mean = df.groupby([cat_col, f'{num_col}_binned'])['roas'].mean()
                    features[f'{cat_col}_{num_col}_combo_target'] = df.apply(
                        lambda row: combo_target_mean.get((row[cat_col], row[f'{num_col}_binned']), df['roas'].mean()),
                        axis=1
                    )

        return features
```

### 4.4 时间-交互特征（50+）

```python
    def extract_temporal_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间-交互特征 (50+)

        包括:
        - 时间窗口内的交互
        - 时间-类别组合
        - 时间趋势-特征交互
        """

        features = pd.DataFrame(index=df.index)
        df['date'] = pd.to_datetime(df['date'])

        # 1. 时间-特征交互 (20)
        numeric_cols = ['impressions', 'spend', 'roas']
        for col in numeric_cols:
            if col not in df.columns:
                continue

            # 工作日 * 特征
            features[f'{col}_weekday'] = df[col] * df['date'].dt.weekday

            # 月初 * 特征
            features[f'{col}_month_start'] = df[col] * (df['date'].dt.day <= 7).astype(int)

            # 周末 * 特征
            features[f'{col}_weekend'] = df[col] * (df['date'].dt.weekday >= 5).astype(int)

        # 2. 时间-类别交互 (15)
        categorical_cols = ['campaign_objective', 'ad_format']
        for cat_col in categorical_cols:
            if cat_col not in df.columns:
                continue

            # 类别 * 小时
            df['hour'] = df['date'].dt.hour
            for category in df[cat_col].unique()[:3]:
                features[f'{cat_col}_{category}_hour'] = ((df[cat_col] == category) * df['hour']).astype(int)

        # 3. 趋势-特征交互 (10)
        for col in ['spend', 'roas']:
            if col not in df.columns:
                continue

            # 趋势 * 当前值
            trend = df[col].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            features[f'{col}_trend_interaction'] = df[col] * trend

            # 波动率 * 当前值
            volatility = df[col].rolling(7).std()
            features[f'{col}_volatility_interaction'] = df[col] * volatility

        # 4. 累积-特征交互 (5)
        features['cumulative_spend_roas'] = df['spend'].cumsum() * df['roas']
        features['cumulative_impressions_ctr'] = df['impressions'].cumsum() * df['ctr']

        return features
```

---

## 📈 Part 5: 全面的时间序列特征工程（150+ 特征）

### 5.1 高级时间序列特征

```python
class AdvancedTimeSeriesFeatures:
    """高级时间序列特征"""

    def extract_advanced_timeseries_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        高级时间序列特征 (100+)

        包括:
        - 时间序列分解
        - 傅里叶变换
        - 小波变换
        - 动态时间规整
        - 时间序列形状特征
        """

        features = pd.DataFrame(index=df.index)
        df = df.sort_values('date')

        for col in ['spend', 'roas', 'impressions']:
            if col not in df.columns:
                continue

            # 1. 时间序列分解 (15)
            if len(df) >= 14:  # 需要足够数据
                try:
                    decomposition = seasonal_decompose(df[col].fillna(0), model='additive', period=7)
                    features[f'{col}_trend'] = decomposition.trend
                    features[f'{col}_seasonal'] = decomposition.seasonal
                    features[f'{col}_residual'] = decomposition.resid

                    # 趋势强度
                    features[f'{col}_trend_strength'] = np.abs(decomposition.trend).rolling(7).mean()

                    # 季节性强度
                    features[f'{col}_seasonal_strength'] = np.abs(decomposition.seasonal).rolling(7).mean()

                    # 残差波动
                    features[f'{col}_residual_volatility'] = decomposition.resid.rolling(7).std()
                except:
                    pass

            # 2. 平稳性检验 (3)
            if len(df) >= 30:
                try:
                    from statsmodels.tsa.stattools import adfuller, kpss
                    result_adf = adfuller(df[col].dropna())
                    features[f'{col}_adf_statistic'] = result_adf[0]
                    features[f'{col}_adf_pvalue'] = result_adf[1]
                    features[f'{col}_is_stationary'] = (result_adf[1] < 0.05).astype(int)
                except:
                    pass

            # 3. 自相关和偏自相关 (10)
            for lag in [1, 2, 3, 7, 14]:
                features[f'{col}_autocorr_lag{lag}'] = df[col].autocorr(lag=lag)

                # Partial autocorrelation
                try:
                    from statsmodels.tsa.stattools import pacf
                    pacf_values = pacf(df[col].dropna(), nlags=lag)
                    features[f'{col}_pacf_lag{lag}'] = pacf_values[lag] if lag < len(pacf_values) else 0
                except:
                    features[f'{col}_pacf_lag{lag}'] = 0

            # 4. 变化点检测 (5)
            features[f'{col}_change_score'] = self._detect_change_points(df[col])
            features[f'{col}_cusum'] = self._cusum_statistic(df[col])
            features[f'{col}_zscore_change'] = np.abs((df[col] - df[col].rolling(30).mean()) / df[col].rolling(30).std())
            features[f'{col}_mean_diff_short_long'] = df[col].rolling(7).mean() - df[col].rolling(30).mean()
            features[f'{col}_ratio_short_long'] = df[col].rolling(7).mean() / (df[col].rolling(30).mean() + 1e-6)

            # 5. 时间序列形状特征 (8)
            features[f'{col}_curve_length'] = self._curve_length(df[col])
            features[f'{col}_zero_crossing_rate'] = self._zero_crossing_rate(df[col])
            features[f'{col}_peak_count'] = self._peak_count(df[col])
            features[f'{col}_trough_count'] = self._trough_count(df[col])
            features[f'{col}_slope_sign_changes'] = self._slope_sign_changes(df[col])
            features[f'{col}_local_maxima'] = df[col].rolling(5, center=True).apply(lambda x: x[2] == max(x))
            features[f'{col}_local_minima'] = df[col].rolling(5, center=True).apply(lambda x: x[2] == min(x))
            features[f'{col}_monotonicity'] = self._monotonicity(df[col])

        return features

    def _detect_change_points(self, series: pd.Series, window: int = 7) -> pd.Series:
        """检测变化点"""
        mean_short = series.rolling(window).mean()
        mean_long = series.rolling(window * 4).mean()
        return np.abs((mean_short - mean_long) / (mean_long + 1e-6))

    def _cusum_statistic(self, series: pd.Series) -> pd.Series:
        """CUSUM 统计量"""
        target = series.rolling(30).mean()
        return (series - target).cumsum()

    def _curve_length(self, series: pd.Series) -> float:
        """曲线长度"""
        diff = series.diff().fillna(0)
        return np.sqrt(1 + diff**2).sum()

    def _zero_crossing_rate(self, series: pd.Series) -> float:
        """零交叉率"""
        centered = series - series.mean()
        return (centered.diff().fillna(0) < 0).astype(int).sum() / len(series)

    def _peak_count(self, series: pd.Series) -> int:
        """峰值数量"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(series.dropna().values)
        return len(peaks)

    def _trough_count(self, series: pd.Series) -> int:
        """谷值数量"""
        from scipy.signal import find_peaks
        troughs, _ = find_peaks(-series.dropna().values)
        return len(troughs)

    def _slope_sign_changes(self, series: pd.Series) -> int:
        """斜率符号变化次数"""
        diff = series.diff().fillna(0)
        sign_changes = ((diff > 0) != (diff.shift(1) > 0)).astype(int)
        return sign_changes.sum()

    def _monotonicity(self, series: pd.Series) -> float:
        """单调性"""
        from scipy.stats import pearsonr
        x = np.arange(len(series))
        correlation, _ = pearsonr(x, series.fillna(0))
        return abs(correlation)
```

### 5.2 频域特征

```python
    def extract_frequency_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        频域特征 (30+)

        包括:
        - FFT 特征
        - 功率谱密度
        - 频谱熵
        - 主频率
        """

        features = pd.DataFrame(index=df.index)

        for col in ['spend', 'roas', 'impressions']:
            if col not in df.columns:
                continue

            # 1. FFT 变换 (8)
            fft_values = np.fft.fft(df[col].fillna(0).values)
            fft_freq = np.fft.fftfreq(len(df))

            # 主频率
            dominant_freq_idx = np.argmax(np.abs(fft_values[1:len(fft_values)//2])) + 1
            features[f'{col}_dominant_freq'] = np.abs(fft_freq[dominant_freq_idx])
            features[f'{col}_dominant_freq_power'] = np.abs(fft_values[dominant_freq_idx])

            # 频谱能量
            features[f'{col}_spectral_energy'] = np.sum(np.abs(fft_values)**2)

            # 频谱质心
            power_spectrum = np.abs(fft_values)**2
            features[f'{col}_spectral_centroid'] = np.sum(fft_freq * power_spectrum) / (np.sum(power_spectrum) + 1e-6)

            # 频谱带宽
            features[f'{col}_spectral_bandwidth'] = np.sqrt(
                np.sum(((fft_freq - features[f'{col}_spectral_centroid'])**2) * power_spectrum) /
                (np.sum(power_spectrum) + 1e-6)
            )

            # 频谱熵
            power_spectrum_norm = power_spectrum / (np.sum(power_spectrum) + 1e-6)
            features[f'{col}_spectral_entropy'] = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm + 1e-6))

            # 低频能量比例
            low_freq_mask = np.abs(fft_freq) < 0.1
            features[f'{col}_low_freq_energy_ratio'] = (
                np.sum(power_spectrum[low_freq_mask]) / (np.sum(power_spectrum) + 1e-6)
            )

            # 高频能量比例
            high_freq_mask = np.abs(fft_freq) > 0.3
            features[f'{col}_high_freq_energy_ratio'] = (
                np.sum(power_spectrum[high_freq_mask]) / (np.sum(power_spectrum) + 1e-6)
            )

        return features
```

### 5.3 时间序列模式特征

```python
    def extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间序列模式特征 (20+)

        包括:
        - 周期性模式
        - 趋势模式
        - 季节性模式
        """

        features = pd.DataFrame(index=df.index)
        df = df.sort_values('date')

        for col in ['spend', 'roas']:
            if col not in df.columns:
                continue

            # 1. 周期性特征 (7)
            # 周内周期性
            df['day_of_week'] = df['date'].dt.dayofweek
            dow_pattern = df.groupby('day_of_week')[col].mean()
            features[f'{col}_dow_pattern_strength'] = dow_pattern.std() / (dow_pattern.mean() + 1e-6)

            # 月内周期性
            df['day_of_month'] = df['date'].dt.day
            dom_pattern = df.groupby('day_of_month')[col].mean()
            features[f'{col}_dom_pattern_strength'] = dom_pattern.std() / (dom_pattern.mean() + 1e-6)

            # 2. 趋势模式 (5)
            # 线性趋势强度
            x = np.arange(len(df))
            slope, intercept = np.polyfit(x, df[col].fillna(0), 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((df[col].fillna(0) - y_pred) ** 2)
            ss_tot = np.sum((df[col].fillna(0) - df[col].mean()) ** 2)
            features[f'{col}_trend_r2'] = 1 - (ss_res / (ss_tot + 1e-6))
            features[f'{col}_trend_slope'] = slope
            features[f'{col}_trend_direction'] = (slope > 0).astype(int)
            features[f'{col}_is_uptrend'] = (slope > 0) & (features[f'{col}_trend_r2'] > 0.5)
            features[f'{col}_is_downtrend'] = (slope < 0) & (features[f'{col}_trend_r2'] > 0.5)

            # 3. 季节性模式 (5)
            # 季节性强度
            detrended = df[col] - df[col].rolling(7).mean()
            features[f'{col}_seasonality_strength'] = detrended.std() / (df[col].std() + 1e-6)

            # 周期检测
            from scipy.signal import find_peaks
            autocorr = [df[col].autocorr(lag=lag) for lag in range(1, 31)]
            peaks, _ = find_peaks(autocorr)
            features[f'{col}_dominant_period'] = peaks[0] if len(peaks) > 0 else 0
            features[f'{col}_has_weekly_seasonality'] = (7 in peaks)
            features[f'{col}_has_monthly_seasonality'] = (30 in peaks)

        return features
```

---

## 📊 Part 6: 特征选择和降维

### 6.1 特征选择方法

```python
class FeatureSelector:
    """特征选择器"""

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'combined',
        n_features: int = 100
    ) -> list:
        """
        特征选择

        方法:
        - variance: 方差阈值
        - correlation: 相关性过滤
        - mutual_info: 互信息
        - chi2: 卡方检验
        - importance: 模型特征重要性
        - combined: 组合方法
        """

        selected_features = []

        if method in ['variance', 'combined']:
            # 1. 方差阈值
            variance_selector = VarianceThreshold(threshold=0.01)
            variance_selector.fit(X)
            selected_features.append(X.columns[variance_selector.get_support()])

        if method in ['correlation', 'combined']:
            # 2. 相关性过滤
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > 0.95)
            ]
            selected_features.append([col for col in X.columns if col not in high_corr_features])

        if method in ['mutual_info', 'combined']:
            # 3. 互信息
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y)
            mi_selected = X.columns[np.argsort(mi_scores)[-n_features:]]
            selected_features.append(mi_selected)

        if method == 'combined':
            # 取交集
            selected = set(selected_features[0])
            for s in selected_features[1:]:
                selected.intersection_update(s)
            return list(selected)

        return selected_features[0]
```

### 6.2 降维技术

```python
class DimensionalityReducer:
    """降维器"""

    def reduce_dimensions(
        self,
        X: pd.DataFrame,
        method: str = 'pca',
        n_components: int = 50
    ) -> pd.DataFrame:
        """
        降维

        方法:
        - pca: 主成分分析
        - tsne: t-SNE
        - umap: UMAP
        - autoencoder: 自编码器
        """

        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(X)

            columns = [f'pc_{i}' for i in range(n_components)]
            return pd.DataFrame(reduced, columns=columns, index=X.index)

        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(X)

            columns = [f'tsne_{i}' for i in range(n_components)]
            return pd.DataFrame(reduced, columns=columns, index=X.index)

        elif method == 'umap':
            from umap import UMAP
            reducer = UMAP(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(X)

            columns = [f'umap_{i}' for i in range(n_components)]
            return pd.DataFrame(reduced, columns=columns, index=X.index)

        return X
```

---

## 📋 附录

### A. 特征名称混淆完整映射示例

```python
# 内部映射表 (config/feature_mappings/v1.0.json)
{
  "数值特征": {
    "f1": "impressions_mean",
    "f2": "clicks_sum",
    "f3": "spend_median",
    "f4": "roas_std",
    "f5": "ctr",
    "f6": "cvr",
    "f7": "cpa",
    "f8": "cpc",
    "f9": "cpm",
    "f10": "roas_7d_avg",
    ...
  },

  "类别特征": {
    "f251": "campaign_objective_encoding",
    "f252": "ad_format_onehot_video",
    "f253": "targeting_gender_onehot_female",
    "f254": "call_to_action_encoding",
    ...
  },

  "时序特征": {
    "f351": "impressions_rolling_mean_7d",
    "f352": "spend_lag_7d",
    "f353": "roas_trend_slope",
    "f354": "clicks_autocorr_lag7",
    ...
  },

  "交互特征": {
    "f401": "impressions_plus_clicks",
    "f402": "spend_times_roas",
    "f403": "objective_format_combo",
    "f404": "impressions_mean_by_objective",
    ...
  }
}
```

### B. 特征统计总览

| 特征类别 | Phase 1 (Python) | Phase 2 (Spark) | Phase 3 (Streaming) |
|---------|------------------|-----------------|-------------------|
| **基础数值** | 50 | 50 | 50 |
| **高级数值** | 60 | 80 | 100 |
| **基础类别** | 20 | 30 | 40 |
| **高级类别** | 30 | 50 | 60 |
| **时序基础** | 40 | 60 | 80 |
| **时序高级** | 30 | 50 | 70 |
| **交互特征** | 100 | 200 | 500 |
| **总计** | **330** | **520** | **900+** |

### C. 技术栈总结

| Phase | 批处理 | 流处理 | 存储 | 调度 |
|-------|--------|--------|------|------|
| **Phase 1** | Pandas/NumPy | - | CSV/Parquet | Cron |
| **Phase 2** | PySpark | - | S3 + Parquet | Airflow |
| **Phase 3** | PySpark | Lambda/Step Functions | Redis + DynamoDB | EventBridge |

---

**文档版本**: 2.0
**最后更新**: 2025-01-29
**维护者**: Data Engineering Team
**特征总数**: 900+
