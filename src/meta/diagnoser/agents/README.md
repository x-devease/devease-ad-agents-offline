# Diagnoser AI Agent Team

> 自动化Detector优化系统 - 7x24小时智能进化

## 🎯 目标

通过5个专门化的AI Agents协同工作，实现Diagnoser detector的**零人工干预**自动化优化。

### 核心价值

- **自动分析**: 分析性能指标，识别优化机会
- **自动实验**: 生成方案，修改参数，运行评估
- **自动验证**: 验证效果，检测副作用
- **自动学习**: 归档历史，避免重复错误

---

## 🏗️ 架构概览

```
Orchestrator (编排器)
    │
    ├── PM Agent ──────→ Memory Agent (知识库)
    │      (分析问题)    (提供历史经验)
    │
    ├── Coder Agent ────→ Reviewer Agent
    │      (修改阈值)      (代码审查)
    │         │
    │         └────────────→ Judge Agent
    │                           (运行评估)
    │                                 │
    └─────────────────────────────────┘
                                      │
                                      ▼
                              Memory Agent
                              (归档实验)
```

---

## 📁 文件结构

```
src/meta/diagnoser/agents/
├── __init__.py                    # Package入口
├── orchestrator.py                # 编排器 (已实现)
├── memory_agent.py                # Memory Agent (已实现)
│
├── prompts/                       # 系统Prompts (2,070行)
│   ├── pm_system_prompt.txt       # PM Agent
│   ├── coder_system_prompt.txt    # Coder Agent
│   ├── reviewer_system_prompt.txt # Reviewer Agent
│   ├── judge_system_prompt.txt    # Judge Agent
│   └── memory_system_prompt.txt   # Memory Agent
│
├── memory/                        # Memory存储
│   ├── storage.py                 # 存储实现
│   ├── experiments/               # 实验记录
│   ├── failures/                  # 失败案例
│   └── patterns/                  # 成功模式
│
├── DESIGN.md                      # 详细设计文档
├── ARCHITECTURE.md                # 架构图和数据流
├── AGENT_SYSTEM_UPDATE.md         # 更新说明
└── README.md                      # 本文件
```

---

## 🤖 5个Agent介绍

### 1. PM Agent (产品经理)

**职责**: 分析metrics → 生成优化方案

**核心知识**:
- 3个detector的详细架构
- 可调参数和优化方向
- 阈值调整指南

**输出**: 实验Spec (参数修改、预期效果、验收标准)

### 2. Coder Agent (代码工程师)

**职责**: 修改detector阈值 → 提交代码

**核心能力**:
- 修改DEFAULT_THRESHOLDS
- 添加注释说明
- 保持向后兼容

**禁止行为**:
- ❌ 硬编码作弊
- ❌ 修改核心逻辑
- ❌ 破坏架构

### 3. Reviewer Agent (代码审查)

**职责**: 审查代码 → 通过/拒绝

**检查维度**:
- 架构一致性 (25%)
- 合规性 (30%) - 零容忍
- 逻辑安全 (25%)
- 代码质量 (20%)

**关键风险**:
- Lookahead Bias (使用未来数据)
- 硬编码 (针对测试集作弊)
- 破坏DEFAULT_THRESHOLDS结构

### 4. Judge Agent (评估裁判)

**职责**: 运行评估 → 验证效果

**评估流程**:
1. 运行scripts/evaluate_*.py
2. 对比baseline和new metrics
3. 检测副作用
4. 做出PASS/FAIL决策

**PASS条件**:
- F1提升 >= 3%
- Precision >= 85%
- 无CRITICAL regression

### 5. Memory Agent (知识库)

**职责**: 存储历史 → 检索案例 → 预警风险

**功能**:
- 存储完整实验记录
- 检索类似实验/失败案例/成功模式
- 预警重复失败、过拟合、性能下降
- 参数影响分析

---

## 🚀 快速开始

### 运行Demo

```bash
# 运行完整优化流程演示
python3 scripts/demo_agent_orchestrator.py
```

**输出示例**:
```
================================================================================
AI Agent Team - Demo Optimization Cycle
================================================================================

--- Phase 1: PM Agent Analysis ---
Current metrics: {"precision": 1.0, "recall": 0.541, "f1_score": 0.702}
Experiment spec: 优化FatigueDetector的recall

--- Phase 2: Coder Agent Implementation ---
✅ Implementation completed: 1 files changed

--- Phase 3: Reviewer Agent ---
✅ Review approved (Score: 85/100)

--- Phase 4: Judge Agent Evaluation ---
Evaluation result: PASS
Metrics lift: f1_score: +5.0%, recall: +10.0%, precision: -2.0%

--- Phase 5: Archive to Memory ---
✅ Archived as exp_FatigueDetector_20250203_204218
```

### 使用Memory Agent

```python
from src.meta.diagnoser.agents import MemoryAgent

# 初始化
memory = MemoryAgent()

# 查询类似实验
result = memory.query(
    query_type="SIMILAR_EXPERIMENTS",
    detector="FatigueDetector",
    context={"tags": ["threshold_tuning", "recall_optimization"]}
)

# 保存实验
experiment_id = memory.save_experiment({
    "detector": "FatigueDetector",
    "spec": {...},
    "implementation": {...},
    "review": {...},
    "evaluation": {...},
    "outcome": "SUCCESS"
})
```

---

## 📊 当前状态

### ✅ 已实现 (v0.1.0)

| 组件 | 状态 | 说明 |
|------|------|------|
| 5个Agent Prompts | ✅ 完成 | 2,070行专业知识 |
| Memory Storage | ✅ 完成 | JSON存储系统 |
| Memory Agent | ✅ 完成 | 查询、存储、预警 |
| Orchestrator | ✅ 完成 | 简化版(模拟agents) |
| Demo Script | ✅ 完成 | 端到端演示 |

### 🚧 进行中

| 组件 | 状态 | 预计时间 |
|------|------|----------|
| LLM集成 | 🚧 计划中 | Week 1-2 |
| 真实代码修改 | 🚧 计划中 | Week 1-2 |
| Git操作 | 🚧 计划中 | Week 3-4 |
| 完整评估集成 | 🚧 计划中 | Week 3-4 |

---

## 📚 文档

| 文档 | 说明 |
|------|------|
| [DESIGN.md](DESIGN.md) | 详细设计文档 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 构构图和数据流 |
| [AGENT_SYSTEM_UPDATE.md](AGENT_SYSTEM_UPDATE.md) | 更新说明 |

---

## 🎯 核心特性

### 1. 小步快跑
- 每次只修改一个参数
- 调整幅度5-10%
- 代码diff < 20行

### 2. 对抗机制
- Coder尝试改进 → Judge验证真实效果
- Reviewer防止作弊 → 击碎幻觉

### 3. Memory驱动
- 查询历史经验 → 避免重复错误
- 提取成功模式 → 加速学习
- 预警风险 → 防止过拟合

### 4. Diagnoser深度集成
- 3个detector完整架构知识
- 评估脚本无缝集成
- 代码模式严格遵循

---

## 🔧 支持的Detector

### FatigueDetector (创意疲劳检测)
```python
DEFAULT_THRESHOLDS = {
    "window_size_days": 21,
    "cpa_increase_threshold": 1.2,      # ← 常优化参数
    "consecutive_days": 1,               # ← 常优化参数
    "min_golden_days": 2,
}
```

### LatencyDetector (响应延迟检测)
```python
DEFAULT_THRESHOLDS = {
    "min_daily_spend": 50,              # ← 常优化参数
    "min_drop_ratio": 0.2,              # ← 常优化参数
    "rolling_window_days": 3,
}
```

### DarkHoursDetector (时段表现检测)
```python
DEFAULT_THRESHOLDS = {
    "target_roas": 2.5,                 # ← 常优化参数
    "cvr_threshold_ratio": 0.2,         # ← 常优化参数
    "min_spend_ratio_daily": 0.10,
}
```

---

## 📈 性能指标

### 单次优化循环时间

| 阶段 | 时间 | 说明 |
|------|------|------|
| PM分析 | 10-30秒 | LLM调用(未来) |
| 代码修改 | 5-10秒 | Edit工具 |
| 代码审查 | 10-30秒 | LLM调用(未来) |
| 评估运行 | 60-120秒 | evaluate脚本 |
| Memory归档 | <1秒 | JSON存储 |
| **总计** | **~2-3分钟** | 完整循环 |

### Memory规模

- 单次实验: ~3-5 KB
- 1000次实验: ~3-5 MB
- 10000次实验: ~30-50 MB

---

## 🛡️ 安全与合规

### 数据安全
- ✅ 无敏感数据
- ✅ 可用git管理
- ✅ 人类可读

### 代码安全
- ✅ Reviewer检测硬编码
- ✅ Judge检测作弊
- ✅ 所有变更可追溯

### 运行安全
- ✅ 小步快跑(低风险)
- ✅ 易回滚(单commit)
- ✅ 人工审核(可选)

---

## 🔮 未来计划

### Phase 1: LLM集成 (Week 1-2)
- [ ] 集成Anthropic API
- [ ] 实现真实LLM调用
- [ ] 替换模拟agents

### Phase 2: 代码集成 (Week 3-4)
- [ ] 真实代码修改(Edit工具)
- [ ] Git操作(commit/rollback)
- [ ] 完整评估集成

### Phase 3: 生产部署 (Month 2+)
- [ ] 7x24小时运行
- [ ] 多detector并行优化
- [ ] 在线学习和自适应

---

## 📝 License

MIT License - 详见项目根目录

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**最后更新**: 2025-02-03
**版本**: v0.1.0
**状态**: Beta - Prompts完成，基础实现就绪
