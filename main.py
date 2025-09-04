#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目统一启动脚本

这是项目的主要启动入口，提供简单易用的启动方式。
支持同步和异步启动模式，自动初始化文档库。
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.conversation_graph.startup import startup_application, async_startup_application
from src.conversation_graph.main_langgraph import create_self_corrective_rag_graph
from src.core.config_manager import ConfigManager

def print_banner():
    """打印项目启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    智能对话系统                              ║
║              基于LangChain和多模态大模型                     ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能对话系统启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                    # 标准启动（默认交互模式）
  python main.py --async            # 异步启动
  python main.py --force-reload     # 强制重新加载文档
  python main.py --rules-dir ./docs # 指定文档目录
  python main.py --no-docs          # 跳过文档初始化
  python main.py --no-interactive   # 禁用交互模式
        """
    )
    
    # 启动选项
    parser.add_argument('--async', action='store_true', 
                       help='使用异步启动模式')
    parser.add_argument('--force-reload', action='store_true', 
                       help='强制重新加载所有文档')
    parser.add_argument('--no-docs', action='store_true', 
                       help='跳过文档初始化')
    parser.add_argument('--interactive', action='store_true', default=True,
                       help='启动后进入交互模式（默认启用）')
    parser.add_argument('--no-interactive', action='store_true', 
                       help='禁用交互模式')
    
    # 配置选项
    parser.add_argument('--rules-dir', type=str, 
                       help='文档目录路径 (默认: ./rules)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    
    try:
        # 构建启动配置
        config = {
            'log_level': args.log_level,
            'async_initialization': getattr(args, 'async'),
            'initialize_documents': not args.no_docs,
            'document_config': {
                'force_reload': args.force_reload,
                'knowledge_docs_path': args.rules_dir or './rules',
                'batch_size': 10
            }
        }
        
        # 如果指定了配置文件，加载配置
        if args.config:
            config_manager = ConfigManager(args.config)
            # 可以在这里合并配置文件中的设置
        
        print(f"启动模式: {'异步' if getattr(args, 'async') else '同步'}")
        print(f"文档初始化: {'跳过' if args.no_docs else '启用'}")
        if not args.no_docs:
            print(f"文档目录: {config['document_config']['knowledge_docs_path']}")
            print(f"强制重载: {'是' if args.force_reload else '否'}")
        print(f"日志级别: {args.log_level}")
        print()
        
        # 执行启动
        if getattr(args, 'async'):
            stats = asyncio.run(async_startup_application(config))
        else:
            stats = startup_application(config)
        
        # 检查启动结果
        if not stats['success']:
            print("\n[失败] 启动失败！")
            if stats['errors']:
                print("错误信息:")
                for error in stats['errors']:
                    print(f"  - {error}")
            sys.exit(1)
        
        print("\n[成功] 系统启动成功！")
        
        # 如果启用交互模式（默认启用，除非使用--no-interactive）
        if args.interactive and not args.no_interactive:
            print("\n进入交互模式...")
            print("输入 'quit' 或 'exit' 退出")
            print("输入 'help' 查看可用命令")
            print("-" * 50)
            
            # 创建对话图
            try:
                rag_graph = create_self_corrective_rag_graph()
                print("对话系统已就绪！")
                
                while True:
                    try:
                        user_input = input("\n用户: ").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            print("再见！")
                            break
                        elif user_input.lower() == 'help':
                            print("""
可用命令:
  help     - 显示此帮助信息
  quit/exit/q - 退出程序
  stats    - 显示启动统计信息
  clear    - 清屏
  
直接输入问题即可开始对话。
                            """)
                            continue
                        elif user_input.lower() == 'stats':
                            print(f"\n启动统计信息:")
                            print(f"  启动耗时: {stats['duration']:.2f} 秒")
                            if stats.get('document_stats'):
                                doc_stats = stats['document_stats']
                                print(f"  处理文件: {doc_stats.get('processed_files', 0)} 个")
                                print(f"  生成分块: {doc_stats.get('total_chunks', 0)} 个")
                            continue
                        elif user_input.lower() == 'clear':
                            os.system('cls' if os.name == 'nt' else 'clear')
                            continue
                        elif not user_input:
                            continue
                        
                        # 处理用户查询
                        print("\n助手: 正在处理您的问题...")
                        
                        try:
                            # 调用对话图处理用户输入
                            result = rag_graph.invoke(user_input)
                            answer = result.get('answer', '抱歉，我无法回答这个问题。')
                            confidence = result.get('confidence_score', 0)
                            
                            print(f"助手: {answer}")
                            
                            # 显示置信度（如果较低则提醒用户）
                            if confidence < 0.7:
                                print(f"\n[提示] 答案置信度: {confidence:.2f} (较低，建议重新表述问题)")
                            
                        except Exception as e:
                            print(f"助手: 处理查询时出现错误: {str(e)}")
                            print("助手: 请尝试重新表述您的问题。")
                        
                    except KeyboardInterrupt:
                        print("\n\n再见！")
                        break
                    except EOFError:
                         print("\n\n再见！")
                         break
                    except Exception as e:
                        print(f"\n处理过程中出现错误: {str(e)}")
                        
            except Exception as e:
                print(f"\n创建对话系统失败: {str(e)}")
                print("系统已启动，但交互模式不可用。")
        
        else:
            print("\n系统已就绪，可以开始使用。")
            print("提示: 默认启用交互模式，使用 --no-interactive 参数可禁用")
    
    except KeyboardInterrupt:
        print("\n\n启动被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n启动过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()