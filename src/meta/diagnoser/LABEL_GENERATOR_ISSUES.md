# 标签生成器问题分析与改进建议

## 🔴 严重问题：阈值不一致

### 问题描述

标签生成器（`label_generator.py`）和实际Detector使用了**不一致的阈值**，导致评估结果不准确。

### 当前状态对比

#### FatigueDetector.DEFAULT_THRESHOLDS (实际使用)
```python
{
    "window_size_days": 23,        # ✅ 已优化
    "golden_min_freq": 1.0,
    "golden_max_freq": 2.5,
    "fatigue_freq_threshold": 3.0,
    "cpa_increase_threshold": 1.15, # ✅ 已优化 (从1.2降低)
    "consecutive_days": 1,          # ✅ 已优化
    "min_golden_days": 1,           # ✅ 已优化 (从2降低)
}
```

#### Label Generator._apply_fatigue_rules() (标签生成)
```python
{
    "window_size": 21,              # ❌ 硬编码，旧值！
    "golden_min_freq": 1.0,
    "golden_max_freq": 2.5,
    "fatigue_freq_threshold": 3.0,
    "cpa_increase_threshold": 1.2,  # ❌ 硬编码，旧值！
    "consecutive_days": 1,
    "min_golden_days": 2,           # ❌ 硬编码，旧值！
}
```

### 问题影响

1. **标签不准确**：生成的ground truth标签与detector实际检测逻辑不匹配
2. **评估失真**：Precision/Recall/F1等指标不能反映真实性能
3. **优化误导**：基于错误标签的优化可能走向错误方向

### 具体差异

| 参数 | Detector | Label Generator | 差异 |
|------|----------|-----------------|------|
| window_size_days | 23 | 21 | **+2天** |
| cpa_increase_threshold | 1.15 | 1.2 | **+4.3%** |
| min_golden_days | 1 | 2 | **+1天** |

---

## 🔧 解决方案

### 方案1：统一阈值源（推荐）

让Label Generator从Detector导入阈值：

```python
# label_generator.py
from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector

def _apply_fatigue_rules(self, data, entity_id):
    # 使用detector的阈值
    thresholds = FatigueDetector.DEFAULT_THRESHOLDS

    window_size = thresholds["window_size_days"]
    consecutive_days = thresholds["consecutive_days"]
    min_golden_days = thresholds["min_golden_days"]
    cpa_increase_threshold = thresholds["cpa_increase_threshold"]
    # ... 其他阈值
```

**优点**：
- ✅ 单一数据源，避免不一致
- ✅ Detector优化时标签自动同步
- ✅ 代码更DRY

**缺点**：
- ⚠️ 需要修改label_generator导入结构

---

### 方案2：配置文件统一

创建共享的阈值配置文件：

```python
# src/meta/diagnoser/config/fatigue_thresholds.py
FATIGUE_THRESHOLDS = {
    "window_size_days": 23,
    "golden_min_freq": 1.0,
    "golden_max_freq": 2.5,
    "fatigue_freq_threshold": 3.0,
    "cpa_increase_threshold": 1.15,
    "consecutive_days": 1,
    "min_golden_days": 1,
}

# Detector使用
from src.meta.diagnoser.config.fatigue_thresholds import FATIGUE_THRESHOLDS
class FatigueDetector:
    DEFAULT_THRESHOLDS = FATIGUE_THRESHOLDS

# Label Generator使用
from src.meta.diagnoser.config.fatigue_thresholds import FATIGUE_THRESHOLDS
```

**优点**：
- ✅ 清晰的配置文件
- ✅ Detector和Label Generator都使用同一配置
- ✅ 易于维护和版本控制

**缺点**：
- ⚠️ 需要重构现有代码
- ⚠️ 增加配置复杂度

---

### 方案3：动态传入阈值（最灵活）

让Label Generator接受detector实例或阈值参数：

```python
def generate_labels(
    data,
    detector,  # 传入detector实例
    method="rule_based"
):
    # 使用detector的阈值
    thresholds = detector.DEFAULT_THRESHOLDS
    # ... 生成标签
```

**优点**：
- ✅ 最大灵活性
- ✅ 可以测试不同阈值配置
- ✅ 支持多detector

**缺点**：
- ⚠️ 需要修改调用接口
- ⚠️ 增加函数签名复杂度

---

## 📊 当前评估结果的可信度

### 由于阈值不一致，当前指标可能偏差：

| 指标 | 当前值 | 可能的真实值 | 说明 |
|------|--------|------------|------|
| Precision | 100% | 95-100% | 可能略有下降 |
| Recall | 58.20% | 50-65% | 可能偏差较大 |
| F1-Score | 73.58% | 68-78% | 综合偏差 |

### 为什么？

1. **window_size差异 (23 vs 21)**
   - Detector使用更大的窗口，更保守
   - Label Generator用更小窗口，可能生成更多"正样本"
   - 结果：Detector漏检增多 → Recall降低

2. **cpa_increase_threshold差异 (1.15 vs 1.2)**
   - Detector用更低阈值，更敏感
   - Label Generator用更高阈值，标签更严格
   - 结果：Detector检测到更多但标签认为不是 → FP增加 → Precision降低

3. **min_golden_days差异 (1 vs 2)**
   - Detector只需1天黄金期即可
   - Label Generator需要2天
   - 结果：Detector条件更宽松 → 更多检测 → 可能增加FP

---

## 🎯 立即行动项

### 优先级1：修复阈值不一致（紧急）

```python
# src/meta/diagnoser/judge/label_generator.py
# 修改_apply_fatigue_rules方法

def _apply_fatigue_rules(self, data, entity_id, detector_instance=None):
    """应用疲劳检测规则 - 使用detector的实际阈值"""
    labels = []

    # 获取detector的阈值
    if detector_instance:
        thresholds = detector_instance.DEFAULT_THRESHOLDS
    else:
        # 后备方案：手动导入
        from src.meta.diagnoser.detectors.fatigue_detector import FatigueDetector
        thresholds = FatigueDetector.DEFAULT_THRESHOLDS

    window_size = thresholds["window_size_days"]
    consecutive_days = thresholds["consecutive_days"]
    min_golden_days = thresholds["min_golden_days"]
    cpa_increase_threshold = thresholds["cpa_increase_threshold"]
    golden_min_freq = thresholds["golden_min_freq"]
    golden_max_freq = thresholds["golden_max_freq"]
    fatigue_freq_threshold = thresholds["fatigue_freq_threshold"]

    # ... 使用这些阈值生成标签
```

### 优先级2：重新运行评估

修复后重新运行：
```bash
python3 src/meta/diagnoser/scripts/evaluate_fatigue.py
```

对比修复前后的指标变化。

### 优先级3：为其他Detector也检查同样问题

- LatencyDetector
- DarkHoursDetector

确保所有detector的标签生成都使用一致的阈值。

---

## 📝 改进计划

### Phase 1: 紧急修复（1小时）
1. 修改`label_generator.py`导入detector阈值
2. 重新运行评估验证
3. 更新文档

### Phase 2: 架构优化（2-3小时）
1. 创建统一的阈值配置文件
2. 重构Detector和Label Generator
3. 添加单元测试确保一致性

### Phase 3: 自动化验证（持续）
1. 添加CI检查：标签阈值必须匹配detector阈值
2. 添加评估报告中的阈值版本信息
3. 定期audit阈值一致性

---

## ⚠️ 重要警告

**当前所有基于rule_based标签的评估结果都可能是不可靠的！**

在修复阈值不一致问题之前：
- ❌ 不要信任当前的Precision/Recall/F1指标
- ❌ 不要基于这些指标做优化决策
- ❌ 不要用这些结果比较不同detector版本

**建议**：
1. 立即修复阈值不一致
2. 重新运行所有评估
3. 使用修复后的结果作为baseline

---

生成时间：2026-02-04
问题发现者：Claude (AI Assistant)
优先级：🔴 P0 - 紧急
