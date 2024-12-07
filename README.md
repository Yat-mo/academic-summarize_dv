# 论文批量总结助手 - 你的学术好帮手 📚

> 还在为阅读大量论文而烦恼吗？让AI来帮你解决！ 🤖

这是一个专为学生和研究人员设计的论文批量总结工具。它能自动处理PDF格式的学术论文，生成清晰的总结和直观的思维导图，帮助你快速掌握论文精髓！

## 为什么选择这个工具？ 🎯

- 🚀 **节省时间**：批量处理论文，几分钟内获得核心内容
- 🧠 **智能总结**：使用GPT-4模型，准确提炼论文要点
- 📊 **可视化**：自动生成思维导图，帮助理解论文结构
- 💾 **多种导出**：支持多种格式导出，方便存档和分享
- 📝 **历史记录**：随时查看过去的总结记录
- 🔍 **差异对比**：支持不同版本总结的对比
- 🎨 **美观界面**：简洁直观的用户界面，使用体验好

## 适用场景 🎓

- 📖 文献综述阅读整理
- 🎯 毕业论文调研
- 📑 课程论文阅读
- 🔬 科研项目文献整理
- 📚 考研文献准备

## 系统要求 🔧

- Python 3.8 或更高版本
- 操作系统：Windows/MacOS/Linux
- 稳定的网络连接

## 快速开始 🚀

### 1. 环境准备

首先克隆项目到本地：
```bash
git clone https://github.com/Yat-mo/academic-summarize_dv
cd academic-summarize_dv
```

### 2. 配置虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows用户:
.venv\Scripts\activate

# MacOS/Linux用户:
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```
2. 在`.env`文件中设置你的OpenAI API密钥：
   ```
   OPENAI_API_KEY=你的密钥
   OPENAI_API_BASE=你的API地址
   ```

3. 配置模型选择（在 config.py 中）：
   ```python
   # 使用 GPT-4o 获取最高质量的总结
   OPENAI_MODEL = "gpt-4o"  # 适用于重要论文或需要深度理解

   # 使用 GPT-4o-mini 获取快速总结
   OPENAI_MODEL = "gpt-4o-mini"  # 适用于日常论文阅读或批量处理
   ```

## 使用指南 📖

### 1. 启动应用

```bash
streamlit run app.py
```

### 2. 使用流程

1. 📤 上传PDF文件（支持单个或多个文件）
2. ⚙️ 选择总结选项：
   - 总结深度（简要/详细）
   - 是否生成思维导图
   - 导出格式选择
3. 🚀 点击"开始处理"
4. ⏳ 等待处理完成
5. 📥 查看结果并下载

## 高级功能 🌟

### 1. 自定义总结模板 📝
在 `prompts.py` 中可以自定义总结模板，适应不同类型的论文。

### 2. 批量导出 📦
支持将多篇论文的总结打包下载，格式包括：
- 📄 PDF格式
- 📝 Word格式
- 🗺️ 思维导图
- 📊 HTML报告

### 3. 历史记录管理 📚
- 查看历史总结记录
- 对比不同版本的总结
- 收藏重要总结

## 使用技巧 💡

1. **选择合适的模型**：
   - 使用 `gpt-4o` 处理重要论文：
     * 更深入的理解和分析
     * 更准确的专业术语解释
     * 更全面的逻辑推理
     * 适合写论文综述或重要研究
   
   - 使用 `gpt-4o-mini` 进行日常阅读：
     * 更快的处理速度
     * 更低的API消耗
     * 适合批量处理和初步筛选
     * 满足一般阅读需求

2. **处理大型PDF**：
   - 建议单次上传不超过10个文件
   - 单个PDF最好不超过100页
   - 可以使用分块处理功能处理大文件

3. **优化总结质量**：
   - 选择合适的总结深度
   - 适当调整提示词模板

4. **节省API使用量**：
   - 使用简要总结模式
   - 合理设置最大字数限制
   - 利用缓存功能

## 常见问题解答 ❓

1. **为什么总结失败了？**
   - 检查PDF是否可选择文本
   - 确认网络连接稳定
   - 验证API密钥是否有效

2. **如何提高总结质量？**
   - 使用高质量的PDF文件
   - 调整config.py中的模型参数
   - 优化提示词模板

3. **支持哪些语言的论文？**
   - 英文论文：完全支持
   - 中文论文：完全支持
   - 其他语言：部分支持

## 参与贡献 🤝

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 更新日志 📅

### v1.0.0 (2024-12)
- 🎉 首次发布
- 🚀 支持批量处理PDF
- 🎨 新增思维导图功能
- 📊 添加多种导出格式

## 许可证 📄

MIT License

## 联系我 📬

- 📧 Email: [creat@duck.com]

---

如果这个工具对你有帮助，请给个Star⭐️支持一下！
