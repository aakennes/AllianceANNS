# 构建、搜索与迁移计划

本文只使用通用变量，不记录账号、主机地址或机器目录。先在自己的运行环境中设置 `REPO_ROOT`、`DATA_ROOT`、`MODEL_ROOT`、`OUTPUT_ROOT`；不要把变量的实际值提交到仓库。

## 当前顺序

1. 完成代码与脚本验收，冻结现有搜索/建图实现。
2. 按 [迁移说明](docs/migration.md) 在目标机器重新编译。
3. 检查现有索引与原始查询，随后完整重建 Arxiv2M / SIFT1M。
4. 同机比较 AllianceANN 和 PipeANNFilter；选择率 2^-1、2^-4、2^-7、2^-10，1/8/16 线程。
5. 验收后再处理后续结构重构和性能优化。

## 环境与入口

```bash
cd "$REPO_ROOT"
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build -j8
ctest --test-dir build --output-on-failure -j1

bash scripts/build_arxiv_disk.sh --data_root "$DATA_ROOT/Arxiv" --out_dir "$OUTPUT_ROOT/arxiv-new" --dry_run
bash scripts/build_sift1m_disk.sh --data_root "$DATA_ROOT/Sift1m" --out_dir "$OUTPUT_ROOT/sift-new" --dry_run
```

四个主 wrapper 支持 `--help` / `--dry_run`；搜索入口为 `scripts/test_arxiv_cluster_shuffle_search.sh` 和 `scripts/test_sift1m_cluster_shuffle_search.sh`。实际运行前明确指定 graph/meta、ML graph/sidecar、query/range/GT、CPU 列表与输出目录。使用与新环境相适配的容器，不在计划中保存登录命令或容器身份。

## 参数与结果

已验证的 191 条配置在 [冻结参数](result/history/frozen/best_params_validated.json)。Alliance 的共同设置为 bits3 / K25 / 图搜索 R50，per-query 动态缓存；ef、beam、静态缓存和总动态缓存按具体预算条目取值。匿名路径占位符需由本机配置映射，不直接执行旧 argv。

重建采用既有 M32 / efc256 / o4 / wp5 / wow_efc256 / wow_passes1 / build_mem_gb1.0 / win_anchor top 预设，不能声称是新机全局最优。已有完成索引不覆盖，重建用独立输出目录。

历史日志仅含 dataset、range 和 results；保留 Recall、QPS、延迟、IO、内存及数值结果行，不包含命令行、登录信息、路径或机器环境。缺失的 range 使用 `not_recorded`，不推测补填。详见 [脱敏说明](docs/redaction.md)。
