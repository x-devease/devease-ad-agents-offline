# 客户广告数据安全访问控制设计

## 📋 问题陈述

### 当前状态
- 原始广告数据存储在 `datasets/{customer}/raw/` 目录下
- 数据以明文 CSV 格式存储
- 所有工程师都可以直接访问
- 缺乏访问控制和审计机制

### 安全要求
- ✅ Production Backend → 可访问原始数据
- ❌ Engineers → 只能访问脱敏/聚合数据
- ✅ 完整审计日志
- ✅ 最小权限原则

### 数据类型
本设计涵盖两种数据场景：
1. **离线客户数据** - 批量CSV文件，定期处理
2. **实时客户数据** - API实时获取，流式处理

---

## 🎯 设计目标

### 离线数据目标
1. **批量加密**：对已存储的CSV文件加密
2. **文件级访问控制**：基于环境限制文件访问
3. **脱敏数据集**：为开发提供安全的测试数据

### 实时数据目标
1. **传输加密**：TLS + 请求签名
2. **运行时加密**：敏感字段在内存中也加密
3. **实时脱敏**：API响应自动脱敏
4. **缓存安全**：加密缓存数据

---

## 🏗️ 整体架构

### 架构总览

```
┌────────────────────────────────────────────────────────────┐
│                      客户数据源                              │
│                 Meta Ads API + Historical Export            │
└────────────────────────────────────────────────────────────┘
                              ↓
              ┌───────────────┴───────────────┐
              │                               │
              ↓                               ↓
    ┌─────────────────┐             ┌─────────────────┐
    │  实时数据流      │             │  离线数据流      │
    │  Real-time      │             │  Offline        │
    │  API Stream     │             │  Batch Export   │
    └─────────────────┘             └─────────────────┘
              │                               │
              ↓                               ↓
    ┌─────────────────┐             ┌─────────────────┐
    │ API Gateway     │             │ ETL Pipeline    │
    │ + TLS           │             │ + Encryption    │
    └─────────────────┘             └─────────────────┘
              │                               │
              ↓                               ↓
    ┌─────────────────┐             ┌─────────────────┐
    │ Runtime         │             │ S3 Encrypted    │
    │ Encryption      │             │ Storage         │
    │ (Memory)        │             │ (*.csv.enc)     │
    └─────────────────┘             └─────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ↓
                    ┌─────────────────┐
                    │   IAM Role      │
                    │   限定访问       │
                    └─────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ↓                                           ↓
┌──────────────────┐                    ┌──────────────────┐
│ Production Env   │                    │  Non-Prod Env    │
│ ✅ KMS Decrypt   │                    │  ❌ No KMS Access│
│ ✅ Raw Data      │                    │  ⚠️ Anonymized   │
└──────────────────┘                    └──────────────────┘
        ↓                                           ↓
┌──────────────────┐                    ┌──────────────────┐
│ Ad Miner Engine  │                    │ Dev/Test Data    │
│ (处理数据)        │                    │ (脱敏数据)        │
└──────────────────┘                    └──────────────────┘
        ↓                                           ↓
┌──────────────────┐                    ┌──────────────────┐
│ Output:          │                    │  同样输出         │
│ - Recommendations│                    │  - 推荐结果       │
│ - Patterns       │                    │  - 模式           │
│ - Real-time API  │                    │  - 报告           │
└──────────────────┘                    └──────────────────┘
```

---

## 📁 Part 1: 离线客户数据安全设计

### 1.1 存储架构

```
datasets/
  {customer}/
    raw/
      encrypted/                    # 加密原始数据（只有 production 可访问）
        ad_data_20250129.csv.enc
        ad_data_20250128.csv.enc
        .kms_key_id                 # KMS Key ID 引用
      anonymized/                   # 脱敏数据（供开发测试使用）
        ad_data_20250129_anon.csv
        anonymization_report.json   # 脱敏验证报告
    features/                       # 特征数据（不包含敏感信息）
      features_20250129.csv
    results/                        # 分析结果（可公开访问）
      recommendations.json
      patterns.json
```

### 1.2 数据加密方案

#### Envelope Encryption 流程

```
明文 CSV 文件
    ↓
生成随机数据密钥 (Data Key - 256 bytes)
    ↓
使用 Data Key 加密文件 (AES-256-GCM)
    ↓
使用 AWS KMS 加密 Data Key
    ↓
存储结构:
  ├── ad_data.csv.enc (加密文件内容)
  ├── ad_data.csv.enc.key (加密的 Data Key)
  └── ad_data.csv.enc.metadata.json (元数据)
      ├── encrypted_data_key: "base64..."
      ├── original_file_hash: "sha256..."
      ├── customer_id: "customer-123"
      ├── encryption_algorithm: "AES-256-GCM"
      └── kms_key_id: "alias/ad-data-encryption"
```

#### 加密组件

```python
class OfflineDataEncryptionManager:
    """离线数据加密管理器"""

    def encrypt_csv_file(
        self,
        input_path: Path,
        output_path: Path,
        customer_id: str
    ) -> dict:
        """
        加密CSV文件

        流程:
        1. 生成随机 Data Key
        2. 使用 Data Key 加密文件内容
        3. 使用 KMS 加密 Data Key
        4. 生成元数据文件
        5. 删除明文文件（可选）
        """
        pass

    def decrypt_csv_file(
        self,
        encrypted_path: Path
    ) -> pd.DataFrame:
        """
        解密CSV文件（仅 production）

        流程:
        1. 读取 metadata 获取加密的 Data Key
        2. 调用 KMS Decrypt
        3. 使用解密后的 Data Key 解密文件
        4. 返回 DataFrame
        """
        if self.environment != "production":
            raise DataEncryptionError(
                "Decryption only allowed in production"
            )
        pass
```

### 1.3 数据脱敏策略

#### 敏感字段配置

| 字段类型 | 示例字段 | 处理方式 | 说明 |
|---------|---------|---------|------|
| **ID 字段** | `ad_id`, `adset_id`, `campaign_id` | SHA-256 哈希 (前16字符) | 保持唯一性，不可逆 |
| **敏感信息** | `customer_email`, `phone` | 删除 | 完全移除 |
| **业务名称** | `ad_name`, `campaign_name` | 掩码 | 替换为 `Ad_{hash[:8]}` |
| **URL** | `image_url`, `video_url` | 保留 | 用于特征提取 |
| **数值指标** | `impressions`, `clicks`, `spend` | 保留 + 噪声 | 添加 ±1% 噪声 |
| **业务指标** | `conversions`, `revenue`, `roas` | 保留 + 噪声 | 添加 ±1% 噪声 |
| **时间戳** | `created_at`, `updated_at` | 保留 | 保持时间序列 |

#### 脱敏验证

```python
class OfflineDataAnonymizer:
    """离线数据脱敏器"""

    def anonymize_csv(
        self,
        input_csv: Path,
        output_csv: Path,
        validate: bool = True
    ) -> dict:
        """
        脱敏 CSV 文件

        Args:
            input_csv: 输入加密文件路径
            output_csv: 输出脱敏文件路径
            validate: 是否验证统计特征保留

        Returns:
            脱敏报告 {
                'total_rows': 10000,
                'anonymized_fields': ['ad_id', 'customer_email'],
                'removed_fields': ['phone'],
                'statistical_drift': {
                    'roas_mean_diff_pct': 0.02,  # < 1%
                    'spend_distribution_ks_test': 0.05  # < 0.1
                }
            }
        """
        pass

    def validate_statistical_properties(
        self,
        original_df: pd.DataFrame,
        anonymized_df: pd.DataFrame
    ) -> dict:
        """
        验证脱敏后的统计特征

        检查:
        1. 数值字段均值偏差 < 1%
        2. 分布形状相似 (Kolmogorov-Smirnov test)
        3. 相关性矩阵保持
        """
        pass
```

### 1.4 访问控制

#### 文件系统权限

```bash
# Production 环境
datasets/{customer}/raw/encrypted/
  └── 权限: prod-backend-role:read, engineer:deny

datasets/{customer}/raw/anonymized/
  └── 权限: prod-backend-role:read, engineer:read

# Development 环境
datasets/{customer}/raw/anonymized/
  └── 权限: developer:read

datasets/{customer}/raw/encrypted/
  └── 权限: 完全拒绝 (IAM Policy + S3 Bucket Policy)
```

#### S3 Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyNonProductionAccessToEncrypted",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::ad-data-bucket/*/raw/encrypted/*",
      "Condition": {
        "StringNotEquals": {
          "aws:PrincipalArn": [
            "arn:aws:iam::account:role/production-backend-role"
          ]
        }
      }
    },
    {
      "Sid": "AllowAnonymizedDataAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::account:role/production-backend-role",
          "arn:aws:iam::account:role/developer-role"
        ]
      },
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::ad-data-bucket/*/raw/anonymized/*"
    }
  ]
}
```

---

## 🔄 Part 2: 实时客户数据安全设计

### 2.1 数据流架构

```
┌─────────────────────────────────────────────────────────┐
│              Meta Ads API (Real-time)                    │
│         - Webhook notifications                          │
│         - Real-time insights                             │
│         - Ad performance updates                         │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              API Gateway + NLB                           │
│         - TLS 1.3                                        │
│         - mTLS (mutual TLS)                              │
│         - IP Whitelist                                   │
│         - Request Signing (HMAC-SHA256)                  │
└─────────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              Real-time Data Service                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Request Validation                           │    │
│  │    - API Key                                    │    │
│  │    - Signature verification                     │    │
│  │    - Rate limiting                              │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 2. Runtime Encryption Layer                     │    │
│  │    - Encrypt sensitive fields before processing │    │
│  │    - In-memory encryption (AES-GCM)            │    │
│  │    - Temporary keys (auto-rotate)              │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 3. Processing Pipeline                          │    │
│  │    - Stream processing (never decrypt)          │    │
│  │    - Encrypted analytics                        │    │
│  │    - Homomorphic encryption (future)           │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 4. Response Filtering                           │    │
│  │    - Auto-anonymize responses                   │    │
│  │    - Field-level security                       │    │
│  │    - Environment-based masking                  │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                      ↓
        ┌─────────────┴─────────────┐
        │                           │
        ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│ Encrypted Cache  │      │ Production DB    │
│ (Redis/ElastiCache)│    │ (Column-level)   │
│ - Encrypted at   │      │ - Always encrypted│
│   rest           │      │   in storage     │
└──────────────────┘      └──────────────────┘
```

### 2.2 实时数据加密

#### 运行时加密（In-Memory Encryption）

```python
class RuntimeEncryptionManager:
    """
    运行时加密管理器
    - 敏感字段在内存中始终保持加密状态
    - 使用短期密钥（每小时轮换）
    - 支持密文计算（如适用）
    """

    # 敏感字段配置
    SENSITIVE_FIELDS = {
        'customer_id': 'encrypted',
        'ad_account_id': 'encrypted',
        'targeting_criteria': 'encrypted',
        'budget_details': 'encrypted',
        # 可公开字段
        'impressions': 'plaintext',
        'clicks': 'plaintext',
        'roas': 'plaintext',
    }

    def __init__(self):
        self.current_key = self._get_or_create_rotation_key()
        self.key_rotation_interval = 3600  # 1 hour

    def encrypt_field(self, field_name: str, value: any) -> str:
        """加密单个字段"""
        if field_name not in self.SENSITIVE_FIELDS:
            return value

        encrypted_value = self._encrypt_with_current_key(value)
        return f"enc:v1:{encrypted_value}"

    def decrypt_field(self, field_name: str, encrypted_value: str) -> any:
        """
        解密字段（仅 production）

        实现安全的密文访问：
        - 记录审计日志
        - 验证调用者权限
        - 限制解密频率
        """
        if self.environment != "production":
            raise AccessDeniedError(
                "Field decryption not allowed in non-production"
            )

        self._audit_logger.log_field_decryption(
            field_name=field_name,
            caller=self._get_caller_identity()
        )

        return self._decrypt_with_current_key(encrypted_value)

    def rotate_key(self):
        """轮换加密密钥"""
        new_key = self._generate_key()
        # 旧密钥保留1小时用于解密旧数据
        self.key_deque.append(new_key)
        if len(self.key_deque) > 2:
            self.key_deque.popleft()
```

#### API 请求签名

```python
class RealtimeAPIClient:
    """
    实时 API 客户端
    - 请求签名防篡改
    - 时间戳防重放
    - 环境隔离
    """

    def sign_request(self, payload: dict) -> dict:
        """
        签名 API 请求

        签名算法:
        HMAC-SHA256(
            timestamp + method + endpoint + body,
            API_SECRET_KEY
        )
        """
        timestamp = int(time.time())
        canonical_request = (
            f"{timestamp}{self.method}{self.endpoint}"
            f"{json.dumps(payload, sort_keys=True)}"
        )

        signature = hmac.new(
            self.api_secret_key.encode(),
            canonical_request.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            'payload': payload,
            'headers': {
                'X-Timestamp': str(timestamp),
                'X-API-Key': self.api_key_id,
                'X-Signature': signature,
                'X-Environment': self.environment
            }
        }

    def validate_response(self, response: dict) -> bool:
        """
        验证 API 响应签名

        防止中间人攻击
        """
        pass
```

### 2.3 实时数据脱敏

#### API 响应自动脱敏

```python
class RealtimeDataFilter:
    """
    实时数据过滤器
    - 根据环境自动脱敏响应
    - 字段级权限控制
    - 性能优化（< 1ms overhead）
    """

    # 环境配置
    FIELD_POLICIES = {
        'production': {
            'ad_id': 'expose',
            'customer_id': 'expose',
            'budget': 'expose',
            'targeting': 'expose',
        },
        'staging': {
            'ad_id': 'hash',
            'customer_id': 'hash',
            'budget': 'mask_range',
            'targeting': 'remove',
        },
        'development': {
            'ad_id': 'hash',
            'customer_id': 'remove',
            'budget': 'mask_range',
            'targeting': 'remove',
        }
    }

    def filter_response(
        self,
        data: dict,
        environment: str,
        user_role: str
    ) -> dict:
        """
        过滤 API 响应

        示例:
        Input (production):
        {
            'ad_id': '238500001',
            'customer_id': 'cust_123',
            'budget': 50000,
            'targeting': {'age': ['18-65'], 'gender': 'all'}
        }

        Output (development):
        {
            'ad_id': 'a3f5e9c2',
            'budget': '50k-100k',
            'targeting': None
        }
        """
        policy = self.FIELD_POLICIES.get(environment, {})

        filtered_data = {}
        for field, value in data.items():
            field_policy = policy.get(field, 'remove')

            if field_policy == 'expose':
                filtered_data[field] = value
            elif field_policy == 'hash':
                filtered_data[field] = self._hash_value(value)
            elif field_policy == 'mask_range':
                filtered_data[field] = self._mask_as_range(value)
            elif field_policy == 'remove':
                continue

        return filtered_data

    def _mask_as_range(self, value: float) -> str:
        """将数值转换为范围"""
        if value < 1000:
            return f"{value//100*100}-{(value//100+1)*100}"
        elif value < 100000:
            return f"{value//1000*1000}k-{(value//1000+1)*1000}k"
        else:
            return f"{value//100000*100}k-{(value//100000+1)*100}k"
```

### 2.4 缓存安全

#### Redis 加密配置

```python
class SecureRedisCache:
    """
    安全的 Redis 缓存
    - 数据加密存储
    - 自动密钥轮换
    - TTL 管理
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST'),
            port=6379,
            ssl=True,  # TLS enabled
            ssl_cert_reqs='required'
        )
        self.encryption_manager = RuntimeEncryptionManager()

    def set(self, key: str, value: dict, ttl: int = 3600):
        """
        加密存储

        存储格式:
        {
            'version': 2,
            'encrypted_data': 'base64(...)',
            'encryption_key_id': 'key-2025-01-29-10',
            'timestamp': 1706523456,
            'ttl': 3600
        }
        """
        encrypted_value = self.encryption_manager.encrypt_field(
            'cache_data',
            json.dumps(value)
        )

        cache_entry = {
            'version': 2,
            'encrypted_data': encrypted_value,
            'encryption_key_id': self.encryption_manager.current_key_id,
            'timestamp': int(time.time()),
            'ttl': ttl
        }

        self.redis_client.setex(
            key,
            ttl,
            json.dumps(cache_entry)
        )

    def get(self, key: str) -> dict:
        """
        解密读取（仅 production）
        """
        raw_value = self.redis_client.get(key)

        if self.environment != "production":
            # Non-production: 返回错误或模拟数据
            raise AccessDeniedError(
                "Cache decryption not allowed in non-production"
            )

        cache_entry = json.loads(raw_value)
        decrypted_value = self.encryption_manager.decrypt_field(
            'cache_data',
            cache_entry['encrypted_data']
        )

        return json.loads(decrypted_value)
```

### 2.5 实时数据访问控制

#### IAM Policy - Production Real-time Service

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RealtimeAPIAccess",
      "Effect": "Allow",
      "Action": [
        "execute-api:Invoke"
      ],
      "Resource": "arn:aws:execute-api:region:account:api-id/*"
    },
    {
      "Sid": "KMSDecryptRuntimeData",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "arn:aws:kms:region:account:key/runtime-data-key-id"
    },
    {
      "Sid": "RedisAccess",
      "Effect": "Allow",
      "Action": [
        "redis:Connect",
        "redis:Get",
        "redis:Set"
      ],
      "Resource": "arn:aws:elasticache:region:account:cluster:redis-cluster"
    }
  ]
}
```

#### IAM Policy - Developer (No Realtime Data Access)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyRuntimeDataDecryption",
      "Effect": "Deny",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "arn:aws:kms:region:account:key/runtime-data-key-id"
    },
    {
      "Sid": "AllowReadOnlyAnonymized",
      "Effect": "Allow",
      "Action": [
        "execute-api:Invoke"
      ],
      "Resource": "arn:aws:execute-api:region:account:api-id/prod/GET/anonymized/*"
    }
  ]
}
```

---

## 📊 Part 3: 统一审计日志系统

### 3.1 审计事件类型

```python
class AuditEventType(Enum):
    """审计事件类型"""

    # 离线数据事件
    OFFLINE_FILE_ENCRYPT = "offline_file_encrypt"
    OFFLINE_FILE_DECRYPT = "offline_file_decrypt"
    OFFLINE_FILE_ACCESS_DENIED = "offline_file_access_denied"
    OFFLINE_ANONYMIZED_DATA_ACCESS = "offline_anonymized_data_access"

    # 实时数据事件
    REALTIME_API_REQUEST = "realtime_api_request"
    REALTIME_FIELD_DECRYPT = "realtime_field_decrypt"
    REALTIME_CACHE_ACCESS = "realtime_cache_access"
    REALTIME_API_RESPONSE_FILTERED = "realtime_api_response_filtered"

    # 密钥管理事件
    KEY_ROTATION = "key_rotation"
    KEY_ACCESS_DENIED = "key_access_denied"
```

### 3.2 审计日志实现

```python
class UnifiedAuditLogger:
    """
    统一审计日志系统
    - 支持离线和实时数据事件
    - 发送到 CloudWatch Logs
    - 集成告警
    """

    def __init__(self):
        self.cloudwatch_client = boto3.client('logs')
        self.log_group = "/aws/data-access/audit"
        self.log_stream = self._get_log_stream_name()

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        resource_id: str,
        environment: str,
        metadata: dict
    ):
        """
        记录审计事件

        Args:
            event_type: 事件类型
            user_id: 用户/服务 ID
            resource_id: 资源标识（文件路径、API endpoint等）
            environment: 环境（production/staging/development）
            metadata: 额外元数据
        """
        log_entry = {
            'event_type': event_type.value,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'resource_id': resource_id,
            'environment': environment,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent(),
            'metadata': metadata
        }

        # 发送到 CloudWatch Logs
        self._send_to_cloudwatch(log_entry)

        # 高敏感事件发送告警
        if self._is_high_sensitivity_event(event_type):
            self._send_alert(log_entry)

    def _is_high_sensitivity_event(self, event_type: AuditEventType) -> bool:
        """判断是否为高敏感事件"""
        high_sensitivity_events = {
            AuditEventType.OFFLINE_FILE_DECRYPT,
            AuditEventType.REALTIME_FIELD_DECRYPT,
            AuditEventType.KEY_ACCESS_DENIED,
        }
        return event_type in high_sensitivity_events

    def _send_alert(self, log_entry: dict):
        """发送告警到 SNS"""
        sns_client = boto3.client('sns')
        sns_client.publish(
            TopicArn=os.getenv('ALERT_SNS_TOPIC'),
            Subject=f"Security Alert: {log_entry['event_type']}",
            Message=json.dumps(log_entry, indent=2)
        )
```

### 3.3 CloudWatch Logs Insights 查询

```sql
-- 查询所有解密事件（离线 + 实时）
fields @timestamp, event_type, user_id, resource_id, environment
| filter event_type like /DECRYPT/
| sort @timestamp desc
| stats count() by user_id, environment

-- 查询访问拒绝事件
fields @timestamp, event_type, user_id, resource_id
| filter event_type like /DENIED/
| sort @timestamp desc

-- 检测异常解密行为（单用户短时间内多次解密）
fields @timestamp, user_id, count(*) as decrypt_count
| filter event_type like /DECRYPT/
| stats sum(decrypt_count) as total_decrypts by user_id
| filter total_decrypts > 100
| sort total_decrypts desc
```

---

## 📝 实施步骤

### Phase 1: 离线数据加密（Week 1-2）

#### Week 1: 基础设施
```bash
# 1. 创建 KMS Keys
aws kms create-key --description "Offline Ad Data Encryption"
aws kms create-alias --alias-name alias/offline-ad-data \
  --target-key-id <key-id>

# 2. 创建 S3 Buckets
aws s3 mb s3://ad-data-encrypted
aws s3 mb s3://ad-data-anonymized

# 3. 配置 Bucket Policies
aws s3api put-bucket-policy --bucket ad-data-encrypted \
  --policy file://bucket-policy-encrypted.json
```

#### Week 2: 代码实现
```bash
# 创建加密模块
mkdir -p src/meta/ad/miner/security
touch src/meta/ad/miner/security/__init__.py
touch src/meta/ad/miner/security/offline_encryption.py
touch src/meta/ad/miner/security/anonymization.py
touch src/meta/ad/miner/security/audit.py

# 运行数据迁移
python scripts/migrate_encrypt_data.py
```

### Phase 2: 实时数据加密（Week 3-4）

#### Week 3: API Gateway Setup
```bash
# 1. 创建 API Gateway
aws apigateway create-rest-api --name "Ad Data Real-time API"

# 2. 配置 mTLS
aws apigateway update-rest-api \
  --rest-api-id <api-id> \
  --patch-operations op=replace,path=/endpointConfiguration/types,value=PRIVATE

# 3. 设置请求验证器
aws apigateway create-request-validator \
  --rest-api-id <api-id> \
  --name "RequestSignatureValidator"
```

#### Week 4: Runtime Encryption
```bash
# 创建实时数据服务
mkdir -p src/meta/ad/miner/realtime
touch src/meta/ad/miner/realtime/__init__.py
touch src/meta/ad/miner/realtime/api.py
touch src/meta/ad/miner/realtime/runtime_encryption.py
touch src/meta/ad/miner/realtime/response_filter.py

# 配置 Redis 缓存
aws elasticache create-replication-group \
  --replication-group-id ad-data-cache \
  --engine redis \
  --cache-node-type cache.t3.medium \
  --num-cache-clusters 2 \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled
```

### Phase 3: 审计和监控（Week 5）

```bash
# 1. 创建 CloudWatch Log Group
aws logs create-log-group --log-group-name /aws/data-access/audit

# 2. 创建 SNS Topic for Alerts
aws sns create-topic --name data-security-alerts

# 3. 配置 CloudWatch Alarms
aws cloudwatch put-metric-alarm \
  --alarm-name excessive-decryption-attempts \
  --alarm-description "Alert on excessive decryption" \
  --metric-name DecryptionCount \
  --namespace DataSecurity \
  --statistic Sum \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold
```

### Phase 4: 部署和验证（Week 6）

```bash
# 1. Production Deployment
./deploy.sh production

# 2. Access Control Validation
python tests/test_access_control.py --environment production
python tests/test_access_control.py --environment development

# 3. Performance Testing
python tests/test_performance.py
```

---

## 🔒 安全考虑

### 3.1 离线数据安全

| 安全措施 | 实现方式 |
|---------|---------|
| **静态加密** | AES-256-GCM + KMS |
| **传输加密** | TLS 1.3 for S3 |
| **密钥管理** | AWS KMS (自动轮换) |
| **访问控制** | IAM Role + S3 Bucket Policy |
| **审计日志** | CloudWatch Logs |
| **数据脱敏** | 哈希 + 删除 + 噪声 |

### 3.2 实时数据安全

| 安全措施 | 实现方式 |
|---------|---------|
| **传输加密** | TLS 1.3 + mTLS |
| **运行时加密** | In-memory AES-GCM |
| **请求签名** | HMAC-SHA256 |
| **缓存加密** | Redis at-rest encryption |
| **响应过滤** | 字段级脱敏 |
| **审计日志** | 实时事件流 |

### 3.3 合规性映射

| 合规标准 | 对应措施 |
|---------|---------|
| **GDPR** | ✅ 加密 + 数据脱敏 + 访问控制 |
| **CCPA** | ✅ 访问控制 + 审计日志 + 数据删除 |
| **SOC 2** | ✅ IAM 权限 + 加密 + 监控 |
| **ISO 27001** | ✅ 全面的安全控制 |

---

## 💰 成本估算

### 离线数据成本

| 项目 | 用量 | 单价 | 月成本 |
|-----|------|------|--------|
| KMS Key (Offline) | 1 Key | $1/月 | $1 |
| S3 Storage (加密) | 1 TB | $0.023/GB | $23 |
| S3 Requests | 100K 请求 | $0.0004/1K | $0.04 |
| CloudWatch Logs | 5 GB | $0.50/GB | $2.50 |

**小计**: ~$26.54/月

### 实时数据成本

| 项目 | 用量 | 单价 | 月成本 |
|-----|------|------|--------|
| KMS Key (Runtime) | 1 Key | $1/月 | $1 |
| API Gateway | 1M 请求 | $3.50/M | $3.50 |
| ElastiCache (Redis) | 1 node (t3.medium) | $50/月 | $50 |
| Lambda (数据处理) | 100K 调用 | $0.20/1M | $0.02 |

**小计**: ~$54.52/月

### 总成本

**总计**: ~$81/月

---

## ✅ 验收标准

### 离线数据验收

- [ ] Production 可正常解密加密 CSV 文件
- [ ] Development/Staging 无法解密原始数据
- [ ] 脱敏数据保留统计特征（均值偏差 < 1%）
- [ ] 所有文件访问都有审计日志

### 实时数据验收

- [ ] API 请求必须通过签名验证
- [ ] 敏感字段在内存中加密
- [ ] Non-production 环境 API 响应自动脱敏
- [ ] Redis 缓存数据加密存储

### 安全验收

- [ ] IAM Policy 符合最小权限原则
- [ ] KMS Key 策略正确配置
- [ ] CloudWatch 日志完整记录
- [ ] 异常访问触发告警

### 性能验收

- [ ] 离线解密 < 100ms per file
- [ ] 实时 API 响应时间增加 < 10ms
- [ ] 缓存加密开销 < 5ms

---

## 🚀 后续优化

### 短期（3个月）

1. **批量加密优化**: 使用多线程加速加密过程
2. **密文查询**: 实现确定性加密支持索引查询
3. **自动脱敏验证**: CI/CD 中自动验证脱敏质量

### 中期（6个月）

1. **同态加密**: 支持密文数值计算
2. **零知识证明**: 验证数据处理不泄露隐私
3. **多区域部署**: 支持数据驻留要求

### 长期（12个月）

1. **机密计算**: 使用 AWS Nitro Enclaves
2. **联邦学习**: 跨客户训练模型不共享原始数据
3. **区块链审计**: 不可篡改的审计日志

---

## 📚 参考资料

### AWS 文档
- [AWS KMS Best Practices](https://docs.aws.amazon.com/kms/latest/developerguide/best-practices.html)
- [S3 Encryption](https://docs.aws.amazon.com/AmazonS3/latest/userguide/serving-encrypted-content.html)
- [API Gateway Security](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-control-access.html)

### 安全标准
- [GDPR Compliance](https://gdpr.eu/)
- [NIST Encryption Standards](https://csrc.nist.gov/projects/lightweight-cryptography)

### 设计模式
- [Envelope Encryption Pattern](https://docs.aws.amazon.com/kms/latest/developerguide/encrypt-data-key.html)
- [Field-Level Encryption](https://aws.amazon.com/blogs/database/field-level-encryption-for-amazon-aurora/)

---

**文档版本**: 2.0
**最后更新**: 2025-01-29
**维护者**: Security Team
**数据类型**: 离线数据 + 实时数据
