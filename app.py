import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import difflib
import zipfile
import tempfile
from datetime import datetime
from utils.pdf_processor import PDFProcessor
from utils.openai_handler import OpenAIHandler
from utils.mindmap_generator import MindmapGenerator
from utils.exporter import PaperExporter
import config

def generate_word_diff(text1, text2):
    matcher = difflib.SequenceMatcher(None, text1, text2)
    result = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            result.append(text1[i1:i2])
        elif opcode == "insert":
            result.append(f"<ins>{text2[j1:j2]}</ins>")
        elif opcode == "delete":
            result.append(f"<del>{text1[i1:i2]}</del>")
        elif opcode == "replace":
            result.append(f"<del>{text1[i1:i2]}</del>")
            result.append(f"<ins>{text2[j1:j2]}</ins>")
    return "".join(result)

def set_page_style():
    st.markdown(
        """
    <style>
        .diff-result {
            font-family: monospace;
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 1.0rem;
        }
        .diff-result ins {
            color: #28a745;
            background-color: #e6ffec;
            text-decoration: none;
        }
        .diff-result del {
            color: #d73a49;
            background-color: #ffeef0;
            text-decoration: line-through;
        }
        
        @media (prefers-color-scheme: dark) {
            .diff-result ins {
                color: #85e89d;
                background-color: transparent;
            }
            .diff-result del {
                color: #f97583;
                background-color: transparent;
            }
        }

        h1, h2, h3 {
            color: #1e3a8a;
        }

        .css-1d391kg {
            padding-top: 1rem;
            padding-right: 0.5rem;
            padding-left: 0.5rem;
        }
        
        .css-1d391kg .block-container {
            padding-top: 1rem;
        }
        
        .css-1q1n0ol {
            max-width: 14rem;
        }
        
        /* 自定义样式 */
        .stProgress > div > div > div > div {
            background-color: #1e3a8a;
        }
        
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #2563eb;
        }
        
        .success-message {
            color: #28a745;
            padding: 0.5rem;
            border-radius: 4px;
            background-color: #e6ffec;
        }
        
        .error-message {
            color: #d73a49;
            padding: 0.5rem;
            border-radius: 4px;
            background-color: #ffeef0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

class PaperSummarizer:
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        st.set_page_config(
            page_title="论文批量总结助手", 
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        set_page_style()
        self.setup_sidebar()
        self.initialize_session_state()
        
    def setup_sidebar(self):
        st.sidebar.title("⚙️ 设置")
        
        # 从环境变量或session_state获取API密钥
        default_api_key = os.getenv("OPENAI_API_KEY", "")
        if "api_key" not in st.session_state:
            st.session_state.api_key = default_api_key
        
        api_key = st.sidebar.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.api_key,
            key="api_key_input"
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # 从环境变量或session_state获取API基础URL
        default_api_base = os.getenv("OPENAI_API_BASE", config.DEFAULT_API_BASE)
        if "api_base" not in st.session_state:
            st.session_state.api_base = default_api_base
        
        api_base = st.sidebar.text_input(
            "API Base URL",
            value=st.session_state.api_base,
            key="api_base_input"
        )
        if api_base != st.session_state.api_base:
            st.session_state.api_base = api_base
        
        # 添加模式选择
        self.summary_style = st.sidebar.radio(
            "💡 总结风格",
            ["学术模式", "通俗模式"],
            help="选择总结的风格"
        )
        
        # 默认设置summary_mode
        self.summary_mode = "标准模式"  # 设置默认值
        
        if self.summary_style == "学术模式":
            self.summary_mode = st.sidebar.radio(
                "📝 详细程度",
                ["简洁模式", "标准模式", "详细模式"],
                help="选择学术程度"
            )
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("""
            ### 模式说明
            
            🔹 **简洁模式**
            - 提供论文的核心观点和主要结论
            - 适合快速了解论文要点
            
            🔹 **标准模式**
            - 包含研究方法、结果和讨论
            - 适合深入理解研究内容
            
            🔹 **详细模式**
            - 深入分析研究背景、方法、结果和影响
            - 适合全面掌握论文内容
            """)
        else:
            self.summary_mode = "通俗模式"  # 设置为通俗模式
            st.sidebar.markdown("---")
            st.sidebar.markdown("""
            ### 通俗模式说明
            
            🔸 **特点**
            - 用通俗易懂的语言解释学术内容
            - 重点关注实际应用价值
            - 通过具体案例说明
            
            🔸 **适用场景**
            - 快速理解论文核心内容
            - 了解研究的实际意义
            - 寻找可能的应用方向
            """)
    
    def initialize_session_state(self):
        if "history" not in st.session_state:
            st.session_state.history = []
        if "processing" not in st.session_state:
            st.session_state.processing = False
            
    def main(self):
        st.title("论文批量总结助手")
        
        uploaded_files = st.file_uploader(
            "上传PDF论文（可多选）", 
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("开始总结"):
            if not st.session_state.api_key:
                st.error("请先设置OpenAI API Key")
                return
                
            st.session_state.processing = True
            
            # 直接处理每个文件
            for file in uploaded_files:
                try:
                    self.process_paper(file)
                except Exception as e:
                    st.error(f"处理失败：{str(e)}")
                    continue
                
            st.session_state.processing = False
            
        self.show_history()
    
    def process_paper(self, file):
        try:
            # 只显示文件名
            st.info(f"正在处理：{file.name}")
            
            # 处理PDF
            processor = PDFProcessor(file)
            total_pages = processor.get_total_pages()
            
            if total_pages == 0:
                st.error(f"无法读取文件：{file.name}")
                return
            
            # 处理文本
            summary_parts = []
            for chunks in processor.process():
                # 调用OpenAI API
                openai_handler = OpenAIHandler(
                    st.session_state.api_key,
                    st.session_state.api_base
                )
                batch_summary = asyncio.run(openai_handler.summarize(
                    chunks,
                    self.summary_mode,
                    self.summary_style
                ))
                summary_parts.append(batch_summary)
            
            # 合并总结
            if len(summary_parts) > 1:
                final_summary = asyncio.run(openai_handler.merge_summaries(
                    summary_parts,
                    self.summary_mode
                ))
            else:
                final_summary = summary_parts[0]
            
            # 生成思维导图
            mindmap_generator = MindmapGenerator()
            try:
                mindmap = mindmap_generator.generate(final_summary)
            except Exception as e:
                st.error(f"思维导图生成失败: {str(e)}")
                mindmap = """
                digraph G {
                    node [shape=box, style=rounded]
                    root [label="论文总结"]
                    error [label="思维导图生成失败"]
                    root -> error
                }
                """
            
            # 保存到历史记录
            st.session_state.history.append({
                "filename": file.name,
                "summary": final_summary,
                "mindmap": mindmap,
                "mode": self.summary_mode,
                "timestamp": pd.Timestamp.now()
            })
            
            st.success(f"完成：{file.name}")
            
        except Exception as e:
            st.error(f"处理失败：{str(e)}")
            
    def download_all_summaries(self):
        """批量下载所有总结（ZIP格式）"""
        if not st.session_state.history:
            st.warning("没有可下载的总结")
            return
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 为每个总结创建单独的文件
            for record in st.session_state.history:
                filename = f"{record['filename']}_{record['timestamp'].strftime('%Y%m%d_%H%M')}.md"
                filepath = os.path.join(temp_dir, filename)
                
                # 建单个总结的内容
                content = f"# {record['filename']}\n\n"
                content += f"## 总结时间：{record['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
                content += f"## 总结模式：{record['mode']}\n\n"
                content += record['summary']
                
                # 写入文件
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # 创建ZIP文件
            zip_filename = f"paper_summaries_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
            zip_filepath = os.path.join(temp_dir, zip_filename)
            
            with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加所有MD文件到ZIP
                for file in os.listdir(temp_dir):
                    if file.endswith('.md'):
                        zipf.write(
                            os.path.join(temp_dir, file),
                            file  # 只保存文件名，不包含路径
                        )
            
            # 读取ZIP文件并通过streamlit下载
            with open(zip_filepath, 'rb') as f:
                zip_data = f.read()
                
            st.download_button(
                label="下载所有总结(ZIP)",
                data=zip_data,
                file_name=zip_filename,
                mime="application/zip"
            )

    def show_history(self):
        if st.session_state.history:
            st.markdown("---")
            col1, col2 = st.columns([8, 2])
            with col1:
                st.header("📋 历史记录")
            with col2:
                self.download_all_summaries()
            
            for idx, record in enumerate(reversed(st.session_state.history)):
                with st.expander(f"📄 {record['filename']} - {record['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    tab1, tab2 = st.tabs([
                        "📝 文本总结",
                        "🔄 思维导图"
                    ])
                    
                    with tab1:
                        st.markdown(
                            f'<div class="diff-result">{record["summary"]}</div>',
                            unsafe_allow_html=True
                        )
                        st.download_button(
                            label="📥 导出Markdown",
                            data=record['summary'],
                            file_name=f"{record['filename']}_{record['timestamp'].strftime('%Y%m%d_%H%M')}.md",
                            mime="text/markdown",
                            key=f"md_{idx}"
                        )
                        
                    with tab2:
                        if record.get('mindmap'):
                            try:
                                with st.spinner("正在渲染思维导图..."):
                                    dot_content = record['mindmap'].strip()
                                    if not dot_content.startswith("digraph"):
                                        st.error("无效的思维导图格式")
                                        st.code(dot_content)
                                    else:
                                        # 显示思维导图
                                        st.graphviz_chart(
                                            dot_content, 
                                            use_container_width=True
                                        )
                                        
                                        # 添加下载按钮
                                        col1, col2 = st.columns([1, 1])
                                        mindmap_generator = MindmapGenerator()
                                        
                                        with col1:
                                            # 下载PNG格式
                                            try:
                                                png_data = mindmap_generator.export_image(dot_content, 'png')
                                                st.download_button(
                                                    label="📥 高清PNG",
                                                    data=png_data,
                                                    file_name=f"{record['filename']}_mindmap_{record['timestamp'].strftime('%Y%m%d_%H%M')}.png",
                                                    mime="image/png",
                                                    key=f"png_{idx}"
                                                )
                                            except Exception as e:
                                                st.error("PNG导出失败")
                                        
                                        with col2:
                                            # 下载DOT源文件
                                            st.download_button(
                                                label="📥 DOT源文件",
                                                data=dot_content,
                                                file_name=f"{record['filename']}_mindmap_{record['timestamp'].strftime('%Y%m%d_%H%M')}.dot",
                                                mime="text/plain",
                                                key=f"dot_{idx}"
                                            )
                            
                            except Exception as e:
                                st.error(f"思维导图显示失败: {str(e)}")
                                with st.expander("查看错误详情"):
                                    st.code(record['mindmap'])
                        else:
                            st.info("未生成思维导图")

if __name__ == "__main__":
    app = PaperSummarizer()
    app.main() 