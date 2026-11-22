## 修改完成情况

| 任务 | 状态 | 说明 |
|------|------|------|
| CMakeLists.txt + index.hpp 声明 | ✅ | 新增编译单元 + ClusterBuildState 前向声明 + 方法签名 |
| SetupClusterBuild + TeardownClusterBuild | ✅ | 按 rank 排序 cluster_members_、旋转全局质心、资源释放 |
| ComputeSubClusters | ✅ | 每 cluster 内 3 轮 Lloyd's KMeans + 旋转子质心 |
| SetupClusterRaBitQ | ✅ | 基于子聚类质心的 1-bit 量化（复用 rabitqlib） |
| BuildLayerWithClusters + RaBitQClusterDist | ✅ | 二分查找窗口 → RaBitQ top-512 heap → 精确距离 → Prune |
| MergeBuild 入口 + build.cpp CLI | ✅ | 条件分支 + `--centroids/--cids/--num_clusters/--sub_clusters/--top_k_exact` |
| 编译通过 + 功能验证 | ✅ | 50k 数据集 recall 与 merge build 持平 |

**修改文件清单：**
- `include/alliance/index/index.hpp` — 新增方法声明 + 成员变量
- `src/alliance/index_cluster_build.cpp` — **新文件**，全部实现
- `src/alliance/index.cpp` — MergeBuild 签名扩展 + Phase 2 条件分支
- `src/example/build.cpp` — CLI 参数解析 + cluster 数据加载
- `src/CMakeLists.txt` — 添加编译单元
- `scripts/test_arxiv_cluster.sh` — **新文件**，端到端测试脚本

---

## 本地测试方法

### 1. 编译

```bash
cd AllianceANNS
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. 小规模快速测试（用 /tmp 已有的 50k 数据）

```bash
# 构建 cluster build 索引
./build/bin/alliance_build \
  --basevec /tmp/alliance_test_50k/base.fvecs \
  --baseatt serial \
  --m 32 --efc 512 --o 2 --wp 0 \
  --threads 8 --space l2 \
  --index_location /tmp/test_cluster.alliance \
  --merge --rabitq_build 1 \
  --wow_thresh 6250 \
  --centroids /tmp/alliance_test_50k/centroids.fvecs \
  --cids /tmp/alliance_test_50k/cids.ivecs \
  --num_clusters 16 --sub_clusters 8 --top_k_exact 512

# 生成 ground truth（如果没有）
./build/bin/alliance_gengt \
  --basevec /tmp/alliance_test_50k/base.fvecs \
  --queryvec /tmp/alliance_test_50k/query.fvecs \
  --att_file serial \
  --query_rng /tmp/alliance_test_50k/query_range.bin \
  --gt_file /tmp/alliance_test_50k/gt.ivecs \
  --k 25 --space l2

# 搜索验证 recall
./build/bin/alliance_search \
  --query_vec /tmp/alliance_test_50k/query.fvecs \
  --query_rng /tmp/alliance_test_50k/query_range.bin \
  --gt_file /tmp/alliance_test_50k/gt.ivecs \
  --k 25 \
  --index_location /tmp/test_cluster.alliance \
  --space l2
```

### 3. 对比测试（cluster vs merge）

不传 `--centroids` 参数即走原有 merge 路径：

```bash
./build/bin/alliance_build \
  --basevec /tmp/alliance_test_50k/base.fvecs \
  --baseatt serial \
  --m 32 --efc 512 --o 2 --wp 0 \
  --threads 8 --space l2 \
  --index_location /tmp/test_merge.alliance \
  --merge --rabitq_build 1 --wow_thresh 6250
```

### 4. Arxiv-2M 全量测试（需要数据在 `${PATH_00003}`）

```bash
bash scripts/test_arxiv_cluster.sh --compare
```

脚本会自动：
1. 检测 centroids/cids 文件是否存在，不存在则调 `scripts/python/ivf.py` 生成
2. 计算 `wow_threshold = N/8`
3. 构建 cluster build 索引 + 搜索 recall
4. `--compare` 额外构建 merge build 对比

### 5. 消融实验

```bash
# 不同 K
bash scripts/test_arxiv_cluster.sh --nc 64 --sub_nc 16
bash scripts/test_arxiv_cluster.sh --nc 256 --sub_nc 32

# 不同 top_k_exact
bash scripts/test_arxiv_cluster.sh --top_k_exact 256
bash scripts/test_arxiv_cluster.sh --top_k_exact 1024

# 不同 wow 覆盖范围
bash scripts/test_arxiv_cluster.sh --wow_frac 16   # N/16
bash scripts/test_arxiv_cluster.sh --wow_frac 32   # N/32

# 小规模切片
bash scripts/test_arxiv_cluster.sh --n_limit 100000 --compare
```

---

### 50k 的测试结果

| 指标 | Cluster Build | Merge Build |
|------|:---:|:---:|
| recall@25 (efs=500) | 0.9896 | 0.9875 |
| recall@25 (efs=100) | 0.9357 | 0.9400 |
| recall@25 (efs=50) | 0.8974 | 0.8948 |
| Phase2 耗时 | 92.9s | 86.4s |

两者 recall 基本持平，功能正确。50k+K=16 时 cluster build 略慢属正常（聚类数太少），大规模数据下（N=2M, K=128~256）cluster build 的 RaBitQ 批量扫描效率优势会体现出来。